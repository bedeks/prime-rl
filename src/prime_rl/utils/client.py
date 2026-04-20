from __future__ import annotations

import asyncio
import os
import socket
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Literal

import httpx
import verifiers as vf
from httpx import AsyncClient
from openai import NotFoundError
from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger


def discover_server_ips(hostname: str) -> list[str]:
    """Discover server IPs via DNS lookup."""
    try:
        _, _, ips = socket.gethostbyname_ex(hostname)
        return sorted(ips)
    except socket.gaierror:
        return []


async def check_server_model(url: str, model_name: str, timeout: float = 5.0) -> tuple[bool, bool]:
    """Check if a server has a specific model loaded."""
    logger = get_logger()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            return model_name in models, len(models) > 0
    except Exception as e:
        logger.debug(f"Failed to check server {url}: {e}")
        return False, False


async def discover_ready_servers(hostname: str, port: int, model_name: str) -> list[str]:
    """Discover servers via DNS that have the requested model loaded."""
    loop = asyncio.get_running_loop()
    ips = await loop.run_in_executor(None, discover_server_ips, hostname)
    if not ips:
        return []

    checks = [check_server_model(f"http://{ip}:{port}", model_name) for ip in ips]
    results = await asyncio.gather(*checks, return_exceptions=True)

    ready_urls = set()
    for ip, result in zip(ips, results):
        if isinstance(result, BaseException):
            continue
        has_model, _ = result
        if has_model:
            ready_urls.add(f"http://{ip}:{port}/v1")

    return sorted(ready_urls)


@dataclass
class AdapterState:
    """State of a LoRA adapter (loaded or desired)."""

    name: str | None = None
    path: Path | None = None
    step: int = 0


ServerStatus = Literal["discovering", "syncing", "ready", "unhealthy"]


@dataclass
class ServerState:
    """State of an individual inference server."""

    ip: str
    url: str
    status: ServerStatus = "discovering"
    loaded_adapter: AdapterState | None = None
    sync_failures: int = 0


class InferencePool:
    """Inference pool for both static and elastic deployments."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions_token",
        eval_client_type: str = "openai_chat_completions",
    ):
        self.logger = get_logger()
        self.client_config = client_config
        self.model_name = model_name
        self.base_model_name = model_name
        self.train_client_type = train_client_type
        self.eval_client_type = eval_client_type
        self._skip_model_check = client_config.skip_model_check
        self._is_elastic = client_config.elastic is not None

        self._train_clients: list[vf.ClientConfig] = []
        self._eval_clients: list[vf.ClientConfig] = []
        self._eval_cycle = cycle(())
        self._static_admin_clients: list[AsyncClient] = []

        self._servers: dict[str, ServerState] = {}
        self._admin_clients: dict[str, AsyncClient] = {}
        self._desired = AdapterState()
        self._client_urls: list[str] = []
        self._eval_index = 0
        self._lock = asyncio.Lock()
        self._sync_task: asyncio.Task[None] | None = None
        self._started = False

        if self._is_elastic:
            elastic_config = client_config.elastic
            if elastic_config is None:
                raise ValueError("Elastic inference pool requires elastic config")
            self.hostname = elastic_config.hostname
            self.port = elastic_config.port
            self.sync_interval = elastic_config.sync_interval
            self.router_url = client_config.router_url
            return

        self.hostname = ""
        self.port = 0
        self.sync_interval = 1.0
        self.router_url = None
        self._train_clients = setup_clients(client_config, client_type=train_client_type)
        self._eval_clients = setup_clients(client_config, client_type=eval_client_type)
        self._eval_cycle = cycle(self._eval_clients)
        self._static_admin_clients = setup_admin_clients(client_config)

    @classmethod
    async def from_config(
        cls,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions_token",
        eval_client_type: str = "openai_chat_completions",
    ) -> InferencePool:
        pool = cls(
            client_config,
            model_name=model_name,
            train_client_type=train_client_type,
            eval_client_type=eval_client_type,
        )
        await pool.start()
        return pool

    @property
    def is_elastic(self) -> bool:
        return self._is_elastic

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        self._rebuild_clients()
        return self._train_clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        if self.is_elastic:
            return [self._admin_clients[ip] for ip in sorted(self._admin_clients)]
        return self._static_admin_clients

    @property
    def eval_clients(self) -> list[vf.ClientConfig]:
        self._rebuild_clients()
        return self._eval_clients

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    async def get_eval_client(self) -> vf.ClientConfig:
        if not self.is_elastic:
            return next(self._eval_cycle)

        while not self.eval_clients:
            await asyncio.sleep(self.sync_interval)
        client = self._eval_clients[self._eval_index % len(self._eval_clients)]
        self._eval_index += 1
        return client

    async def wait_for_ready(self, model_name: str, timeout: int = 1800, min_servers: int = 1) -> None:
        if not self.is_elastic:
            await check_health(self.admin_clients, timeout=timeout)
            await maybe_check_has_model(self.admin_clients, model_name, skip_model_check=self._skip_model_check)
            return

        start = time.time()
        while time.time() - start < timeout:
            await self.sync()
            if self.num_ready_servers >= min_servers:
                return
            self.logger.debug(f"Waiting for servers: {self.num_ready_servers}/{min_servers} ready")
            await asyncio.sleep(self.sync_interval)

        raise TimeoutError(f"Timed out waiting for {min_servers} ready servers (got {self.num_ready_servers})")

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        if not self.is_elastic:
            await update_weights(self.admin_clients, weight_dir, lora_name=lora_name, step=step)
            return

        if lora_name is None:
            raise ValueError("Elastic inference pool requires LoRA training (lora_name must be set)")
        await self.sync_weights(weight_dir, lora_name, step)

    def get_metrics(self) -> dict[str, float]:
        if not self.is_elastic:
            return {}
        return {
            "elastic/num_servers": self.num_servers,
            "elastic/num_ready_servers": self.num_ready_servers,
            "elastic/desired_step": self._desired.step,
        }

    async def stop(self) -> None:
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        if self.is_elastic:
            for ip in list(self._servers):
                await self._remove_server(ip)
            self._client_urls = []
            self._started = False
        else:
            await asyncio.gather(*[client.aclose() for client in self._static_admin_clients])
            self._static_admin_clients = []

        self._train_clients = []
        self._eval_clients = []
        self._eval_cycle = cycle(())
        self._eval_index = 0

    @property
    def ready_urls(self) -> list[str]:
        if not self.is_elastic:
            return list(self.client_config.base_url)
        return [self._build_inference_url(ip) for ip in sorted(self._servers) if self._servers[ip].status == "ready"]

    @property
    def num_servers(self) -> int:
        return len(self._servers)

    @property
    def num_ready_servers(self) -> int:
        return sum(1 for server in self._servers.values() if server.status == "ready")

    async def start(self) -> None:
        if not self.is_elastic or self._started:
            return

        self.logger.debug(f"Starting elastic inference pool (hostname={self.hostname})")
        await self.sync()
        self.logger.debug(f"Initial discovery: {self.num_servers} server(s), {self.num_ready_servers} ready")
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._started = True

    async def sync(self) -> tuple[int, int]:
        if not self.is_elastic:
            return 0, 0

        async with self._lock:
            loop = asyncio.get_running_loop()
            discovered_ips = set(await loop.run_in_executor(None, discover_server_ips, self.hostname))
            known_ips = set(self._servers)

            added = 0
            removed = 0

            for ip in sorted(discovered_ips - known_ips):
                if await self._add_server(ip):
                    added += 1

            for ip in sorted(known_ips - discovered_ips):
                await self._remove_server(ip)
                removed += 1

            for ip in sorted(self._servers):
                if ip not in self._admin_clients:
                    continue
                if not await self._check_server_health(self._admin_clients[ip], ip):
                    self.logger.debug(f"Server {ip} failed health check, removing")
                    await self._remove_server(ip)
                    removed += 1
                elif self._servers[ip].status != "ready":
                    await self._sync_server_adapter(ip)

            return added, removed

    async def sync_weights(self, weights_path: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        async with self._lock:
            self._desired = AdapterState(
                name=lora_name,
                path=weights_path if lora_name else None,
                step=step,
            )
            await asyncio.gather(*[self._sync_server_adapter(ip) for ip in self._servers])

    def _rebuild_clients(self) -> None:
        if not self.is_elastic:
            return

        if self.router_url and self.ready_urls:
            urls = [self.router_url]
        else:
            urls = self.ready_urls

        if urls == self._client_urls:
            return

        self._client_urls = urls
        self._eval_index = 0
        url_config = ClientConfig(
            timeout=self.client_config.timeout,
            connect_timeout=self.client_config.connect_timeout,
            base_url=urls,
            api_key_var=self.client_config.api_key_var,
            headers=self.client_config.headers,
            dp_rank_count=self.client_config.dp_rank_count,
            extra_headers_from_state=self.client_config.extra_headers_from_state,
        )
        self._train_clients = setup_clients(url_config, client_type=self.train_client_type) if urls else []
        self._eval_clients = setup_clients(url_config, client_type=self.eval_client_type) if urls else []

    def _build_url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}"

    def _build_inference_url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}/v1"

    async def _create_admin_client(self, ip: str) -> AsyncClient:
        url = self._build_url(ip)
        config = ClientConfig(
            timeout=self.client_config.timeout,
            connect_timeout=self.client_config.connect_timeout,
            base_url=[f"{url}/v1"],
            api_key_var=self.client_config.api_key_var,
            headers=self.client_config.headers,
        )
        return setup_admin_clients(config)[0]

    async def _get_loaded_adapter(self, ip: str) -> AdapterState | None:
        if ip not in self._admin_clients:
            return None

        try:
            admin = self._admin_clients[ip]
            response = await admin.get("/v1/models")
            response.raise_for_status()
            data = response.json()

            for model in data.get("data", []):
                parent = model.get("parent")
                model_id = model.get("id", "")

                if self._desired.name:
                    if model_id != self._desired.name:
                        continue
                elif not parent:
                    continue
                elif parent != self.model_name:
                    continue

                root = model.get("root", "")
                path = Path(root)
                try:
                    step_part = path.name
                    if step_part.startswith("step_"):
                        step = int(step_part.split("_")[1])
                    elif step_part.startswith("step-"):
                        step = int(step_part.split("-")[1])
                    else:
                        step = 0
                except (ValueError, IndexError):
                    step = 0
                return AdapterState(name=model_id, path=path, step=step)

            self.logger.debug(f"No matching adapter found on {ip} for desired={self._desired.name}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to query /v1/models on {ip}: {e}")
            return None

    def _adapter_matches_desired(self, loaded: AdapterState | None) -> bool:
        if self._desired.path is None:
            return True
        if loaded is None:
            return False
        if loaded.path == self._desired.path:
            return True
        if self._desired.step > 0 and loaded.step == self._desired.step:
            return True
        return False

    async def _sync_server_adapter(self, ip: str) -> bool:
        server = self._servers.get(ip)
        if server is None:
            return False

        loaded = await self._get_loaded_adapter(ip)
        server.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            server.status = "ready"
            return True

        self.logger.debug(
            f"Pre-check failed on {ip}: loaded={loaded.path if loaded else None} "
            f"(step={loaded.step if loaded else None}), desired={self._desired.path} (step={self._desired.step})"
        )
        server.status = "syncing"

        if self._desired.name and self._desired.path:
            try:
                self.logger.debug(f"Loading adapter {self._desired.name} on {ip}")
                await load_lora_adapter([self._admin_clients[ip]], self._desired.name, self._desired.path)
            except Exception as e:
                server.status = "unhealthy"
                server.sync_failures += 1
                self.logger.error(f"Failed to sync server {ip}: {e}")
                return False

        loaded = await self._get_loaded_adapter(ip)
        server.loaded_adapter = loaded

        if self._adapter_matches_desired(loaded):
            server.status = "ready"
            server.sync_failures = 0
            self.logger.debug(f"Successfully synced server {ip}")
            return True

        self.logger.warning(
            f"Adapter mismatch on {ip} after loading: "
            f"loaded={loaded.path if loaded else None} (step={loaded.step if loaded else None}), "
            f"desired={self._desired.path} (step={self._desired.step})"
        )
        server.status = "unhealthy"
        server.sync_failures += 1
        return False

    async def _check_server_health(self, admin_client: AsyncClient, ip: str) -> bool:
        try:
            response = await admin_client.get("/health")
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Server {ip} health check failed: {e}")
            return False

        if self._skip_model_check:
            return True

        try:
            response = await admin_client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            if self.base_model_name not in models:
                self.logger.debug(f"Server {ip} does not have base model {self.base_model_name}")
                return False
        except Exception as e:
            self.logger.debug(f"Server {ip} model check failed: {e}")
            return False

        return True

    async def _add_server(self, ip: str) -> bool:
        try:
            admin_client = await self._create_admin_client(ip)
        except Exception as e:
            self.logger.debug(f"Failed to create admin client for {ip}: {e}")
            return False

        if not await self._check_server_health(admin_client, ip):
            await admin_client.aclose()
            return False

        self.logger.debug(f"Discovered new inference server: {ip}")
        self._admin_clients[ip] = admin_client
        self._servers[ip] = ServerState(ip=ip, url=self._build_url(ip), status="discovering")
        await self._sync_server_adapter(ip)
        return True

    async def _remove_server(self, ip: str) -> None:
        self.logger.debug(f"Inference server removed: {ip}")
        self._servers.pop(ip, None)
        if ip in self._admin_clients:
            await self._admin_clients.pop(ip).aclose()

    async def _sync_loop(self) -> None:
        while True:
            try:
                added, removed = await self.sync()
                if added > 0 or removed > 0:
                    self.logger.debug(
                        f"Elastic pool sync: +{added} -{removed} servers "
                        f"(total: {self.num_servers}, ready: {self.num_ready_servers})"
                    )
            except Exception as e:
                self.logger.error(f"Error in elastic sync loop: {e}")
            await asyncio.sleep(self.sync_interval)


async def setup_inference_pool(
    client_config: ClientConfig,
    model_name: str,
    train_client_type: str = "openai_chat_completions_token",
    eval_client_type: str = "openai_chat_completions",
) -> InferencePool:
    """Create an inference pool from config."""
    logger = get_logger()

    if train_client_type == "openai_chat_completions_token":
        logger.warning(
            "Token-in-token-out (TITO) client is enabled for training. Only use "
            "this if your environment has a linear history and the chat "
            "template has the extension property."
        )

    if client_config.elastic is not None:
        logger.info(
            f"Initializing inference pool (mode=elastic, hostname={client_config.elastic.hostname}, "
            f"port={client_config.elastic.port}, router_url={client_config.router_url})"
        )
    else:
        logger.info(
            f"Initializing inference pool (mode=static, base_url={', '.join(client_config.base_url)}, "
            f"dp_rank_count={client_config.dp_rank_count}, "
            f"api_key_var={client_config.api_key_var}, headers={client_config.headers})"
        )

    return await InferencePool.from_config(
        client_config,
        model_name=model_name,
        train_client_type=train_client_type,
        eval_client_type=eval_client_type,
    )


def setup_clients(client_config: ClientConfig, client_type: str = "openai_chat_completions") -> list[vf.ClientConfig]:
    clients = []
    client_idx = 0
    for base_url in client_config.base_url:
        for dp_rank in range(client_config.dp_rank_count):
            headers = client_config.headers.copy()
            if client_config.dp_rank_count > 1:
                headers["X-data-parallel-rank"] = str(dp_rank)
            clients.append(
                vf.ClientConfig(
                    client_idx=client_idx,
                    client_type=client_type,
                    api_base_url=base_url,
                    api_key_var=client_config.api_key_var,
                    timeout=client_config.timeout,
                    connect_timeout=client_config.connect_timeout,
                    max_connections=8192,
                    max_keepalive_connections=8192,
                    max_retries=10,
                    extra_headers=headers,
                    extra_headers_from_state=client_config.extra_headers_from_state,
                )
            )
            client_idx += 1
    return clients


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set, uses those URLs instead of base_url, allowing
    weight updates to bypass routers in disaggregated P/D deployments.
    """
    urls = client_config.admin_base_url if client_config.admin_base_url else client_config.base_url

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        headers = client_config.headers.copy()
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )

    return [_setup_admin_client(base_url) for base_url in urls]


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[admin_client.get("/v1/models") for admin_client in admin_clients])
    for admin_client, result in zip(admin_clients, results):
        models = result.json()["data"]
        if not any(model["id"] == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin_client.get("/health")
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


NCCL_READY_MARKER = "NCCL_READY"


async def _pause_engines(admin_clients: list[AsyncClient]) -> None:
    """Pause all inference engines, waiting for in-flight requests to drain."""
    logger = get_logger()
    logger.info("Pausing inference engines for weight update")

    async def _pause(client: AsyncClient) -> None:
        response = await client.post("/pause", params={"mode": "keep", "clear_cache": "false"})
        response.raise_for_status()

    await asyncio.gather(*[_pause(client) for client in admin_clients])
    logger.info("All inference engines paused")


async def _resume_engines(admin_clients: list[AsyncClient]) -> None:
    """Resume all inference engines after weight update."""
    logger = get_logger()

    async def _resume(client: AsyncClient) -> None:
        response = await client.post("/resume")
        response.raise_for_status()

    await asyncio.gather(*[_resume(client) for client in admin_clients])
    logger.info("All inference engines resumed")


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
) -> None:
    """Update weights on static inference servers.

    Pauses all engines first to drain in-flight requests, then performs the
    weight update, then resumes. This ensures all DP workers are idle and can
    participate in the collective weight transfer.

    Note: The server-side /update_weights endpoint automatically resets the prefix cache
    to invalidate any cached KV states computed with the old weights.
    """
    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir)
    else:

        async def _update_weights(admin_client: AsyncClient, weight_dir: str | None) -> None:
            response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir})
            response.raise_for_status()

        await _pause_engines(admin_clients)

        try:
            if weight_dir is not None:
                nccl_ready_file = weight_dir / NCCL_READY_MARKER
                nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
                nccl_ready_file.touch()
                logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

            await asyncio.gather(*[_update_weights(admin_client, weight_dir_posix) for admin_client in admin_clients])
        finally:
            await _resume_engines(admin_clients)


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in (404, 500)
    if isinstance(exception, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt and total bounds for `/load_lora_adapter`. A LoRA load is fast
# (small adapter file + KV cache reset, single-digit seconds in practice) but
# the global admin AsyncClient uses `timeout=None`, so a stuck server hangs
# the orchestrator forever inside `InferencePool._sync_server_adapter`.
# `_PER_ATTEMPT` converts a hang into a TimeoutException so tenacity retries;
# `_TOTAL` is the wall-clock budget across all retries — pick whichever
# stop condition fires first.
LORA_LOAD_READ_TIMEOUT_S = 30.0
LORA_LOAD_TOTAL_TIMEOUT_S = 120.0


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper endpoint that also resets the prefix cache to invalidate
    KV states computed with old weights.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = lora_path.as_posix()

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        response = await admin_client.post(
            "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path_posix},
            timeout=httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0),
        )
        response.raise_for_status()

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        await admin_client.post("/v1/unload_lora_adapter", json={"lora_name": lora_name})

    await asyncio.gather(*[_unload_lora_adapter(admin_client) for admin_client in admin_clients])


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int | None = None,
    quantize_in_weight_transfer: bool = False,
) -> None:
    """Initialize NCCL broadcast on all inference servers.

    Each admin client represents one vLLM server. The function computes
    per-server rank_offset and gpus_per_server so that every inference GPU
    gets a unique rank in the NCCL broadcast group.
    """
    logger = get_logger()

    if inference_world_size is None:
        inference_world_size = len(admin_clients)
        logger.warning(
            f"inference_world_size not provided, defaulting to {inference_world_size} (one GPU per admin client)"
        )

    gpus_per_server = inference_world_size // len(admin_clients)

    logger.info(
        f"Initializing NCCL broadcast: {len(admin_clients)} servers, "
        f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
    )

    async def _init_nccl_broadcast(admin_client: AsyncClient, rank_offset: int) -> None:
        try:
            response = await admin_client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "rank_offset": rank_offset,
                    "inference_world_size": inference_world_size,
                    "timeout": timeout,
                    "quantize_in_weight_transfer": quantize_in_weight_transfer,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, client_num * gpus_per_server)
            for client_num, admin_client in enumerate(admin_clients)
        ]
    )
