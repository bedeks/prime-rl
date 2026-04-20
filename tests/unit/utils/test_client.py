import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig, ElasticConfig
from prime_rl.utils.client import InferencePool, _is_retryable_lora_error, load_lora_adapter, setup_inference_pool
from prime_rl.utils.elastic import ElasticInferencePool


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_load_lora_adapter_retries_on_404_then_succeeds():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 404
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=error_response)
        return success_response

    mock_client.post = mock_post

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert call_count == 2


def test_load_lora_adapter_raises_non_retryable_error_immediately():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 400
    mock_client.post.side_effect = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=error_response)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert exc_info.value.response.status_code == 400
    assert mock_client.post.call_count == 1


def test_setup_inference_pool_returns_unified_static_pool():
    client_config = ClientConfig(base_url=["http://localhost:8000/v1"], api_key_var="TEST_API_KEY")

    pool = asyncio.run(setup_inference_pool(client_config, model_name="base-model"))

    assert isinstance(pool, InferencePool)
    assert pool.is_elastic is False

    asyncio.run(pool.stop())


def test_setup_inference_pool_returns_unified_elastic_pool():
    client_config = ClientConfig(
        base_url=["http://localhost:8000/v1"],
        api_key_var="TEST_API_KEY",
        elastic=ElasticConfig(hostname="elastic.test", port=8000, sync_interval=1.0),
    )

    with patch("prime_rl.utils.client.InferencePool.start", new_callable=AsyncMock) as start_mock:
        pool = asyncio.run(setup_inference_pool(client_config, model_name="base-model"))

    assert isinstance(pool, InferencePool)
    assert pool.is_elastic is True
    start_mock.assert_awaited_once()


def test_elastic_inference_pool_is_compatibility_alias():
    assert ElasticInferencePool is InferencePool
