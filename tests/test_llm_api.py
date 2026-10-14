import importlib.util
import base64
import io
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


def _load_module():
    module_name = "easy_media_llm_api_for_tests"
    sys.modules.pop(module_name, None)
    module_path = Path(__file__).parents[1] / "utils" / "llm_api.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _png_data_uri(width=256, height=256):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_minimax_length_to_seconds_rounds_to_nearest_and_clamps():
    module = _load_module()

    assert module.minimax_length_to_seconds(1) == 4
    assert module.minimax_length_to_seconds(49) == 4
    assert module.minimax_length_to_seconds(50) == 4
    assert module.minimax_length_to_seconds(124) == 5
    assert module.minimax_length_to_seconds(10_000) == 15


def test_prompt_enhancer_model_options_match_node_contract():
    module = _load_module()

    assert module.PROMPT_ENHANCER_MODELS == [
        "h3-context-ir (海螺官方)",
        "doubao-seed-2-0-pro-260215 (火山引擎)",
        "glm-5v-turbo (智谱)",
        "bytedance/doubao-seed-2.0-pro (RunningHub)",
        "glm-5v-turbo (RunningHub)",
        "llama.cpp (本地)",
        "openai_compatible (local)",
    ]


def test_third_party_video_url_capabilities_match_provider_documentation():
    module = _load_module()

    assert module.prompt_enhancer_supports_video_url(module.VOLCENGINE_MODEL) is True
    assert module.prompt_enhancer_supports_video_url(module.ZHIPU_MODEL) is True
    assert module.prompt_enhancer_supports_video_url(module.RUNNINGHUB_DOUBAO_MODEL)
    assert module.prompt_enhancer_supports_video_url(module.RUNNINGHUB_GLM_MODEL)


def test_llamacpp_max_tokens_widget_limits():
    module = _load_module()

    assert module.PROMPT_ENHANCER_MAX_TOKENS[module.LLAMACPP_MODEL] == (512, 768)


def _enhance_strata(client, **kwargs):
    options = dict(
        system_prompt="Enhance the video prompt.", user_prompt="A rainy street.",
        task_type="t2v", duration=5, ratio="16:9", seed=42,
    )
    options.update(kwargs)
    return client.enhance(**options)


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:8080", "http://127.0.0.1:8080/v1/",
    "http://127.0.0.1:8080/v1/chat/completions",
])
def test_strata_normalizes_url_and_unloads_without_an_api_key(tmp_path, monkeypatch, base_url):
    module = _load_module()
    monkeypatch.delenv("OPENAI_LOCAL_API_KEY", raising=False)
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/unload?wait=1"):
            return _Response({"status": "unloaded", "loaded": False})
        return _Response({"choices": [{"message": {"content": "```text\nEnhanced\n```"}}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=True, base_url=base_url,
        config_path=tmp_path / "missing.yaml", opener=opener, timeout=600.0,
    )
    assert _enhance_strata(client).prompt == "Enhanced"
    assert [r.full_url for r, _ in requests] == [
        "http://127.0.0.1:8080/v1/chat/completions",
        "http://127.0.0.1:8080/v1/unload?wait=1",
    ]
    assert all(r.get_header("Authorization") is None for r, _ in requests)
    assert all(t == 600.0 for _, t in requests)
    assert json.loads(requests[-1][0].data) == {}
    payload = json.loads(requests[0][0].data)
    assert payload["model"] == "strata"
    assert payload["stream"] is False
    assert payload["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in payload
    assert "max_tokens" not in payload
    assert payload["seed"] == 42
    assert "force_unload" not in payload


def test_strata_sends_custom_parameters_and_images_without_unloading(tmp_path):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return _Response({"choices": [{"message": {"content": "Enhanced", "reasoning_content": "Thinking"}}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "local-key", model_name="strata", base_url="https://host.test/proxy/v1",
        force_unload=False, temperature=0.4, reasoning_effort="low",
        config_path=tmp_path / "missing.yaml", opener=opener,
    )
    assert _enhance_strata(client, task_type="i2v", image_urls=[_png_data_uri()], max_tokens=8192).prompt == "Enhanced"
    assert len(requests) == 1
    assert requests[0].full_url == "https://host.test/proxy/v1/chat/completions"
    assert requests[0].get_header("Authorization") == "Bearer local-key"
    payload = json.loads(requests[0].data)
    assert payload["model"] == "strata"
    assert payload["max_tokens"] == 8192
    assert {key: payload[key] for key in client.local_parameters} == {
        "temperature": 0.4, "reasoning_effort": "low",
    }
    assert any(item["type"] == "image_url" for item in payload["messages"][-1]["content"])


def _strata_context_error(module, request, requested, available, status=400):
    message = (
        f"request exceeds the context window: prompt (7480 tokens) + max tokens ({requested}) "
        f"exceeds the context ({7480 + available + 8}); requests are never truncated. "
        f'Send a smaller max_tokens (at most {available} here), or add "fit_max_tokens": true '
        "to the model's strata-<model>.json to shorten it to the room left (#545)."
    )
    return module.urllib.error.HTTPError(
        request.full_url, status, "failed", {},
        io.BytesIO(json.dumps({"error": {"message": message}}).encode()),
    )


@pytest.mark.parametrize("force_unload", [False, True])
@pytest.mark.parametrize(("requested", "available"), [(131072, 123584), (8192, 2048)])
def test_strata_retries_explicit_output_limit_without_truncating_inputs(
    tmp_path, caplog, force_unload, requested, available,
):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append((request.full_url, json.loads(request.data)))
        if "/unload" in request.full_url:
            return _Response({"status": "unloaded"})
        if len(requests) == 1:
            raise _strata_context_error(module, request, requested, available)
        return _Response({"choices": [{"message": {
            "content": "Final answer", "reasoning_content": "Private thinking",
        }}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=force_unload,
        config_path=tmp_path / "missing.yaml", opener=opener,
    )
    result = _enhance_strata(
        client, max_tokens=requested, task_type="i2v", image_urls=[_png_data_uri()],
    )
    assert result.prompt == "Final answer"
    assert len(requests) == (3 if force_unload else 2)
    assert requests[1] == (requests[0][0], {**requests[0][1], "max_tokens": available})
    assert requests[0][1]["max_tokens"] == requested
    assert f"Reducing max_tokens from {requested} to {available}" in caplog.text
    if force_unload:
        assert requests[-1][0].endswith("/v1/unload?wait=1")


@pytest.mark.parametrize(("status", "available"), [(500, 2048), (400, 0), (400, 8192), (400, 9000)])
def test_strata_does_not_retry_without_a_smaller_positive_allowance(tmp_path, status, available):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request.full_url)
        if "/unload" in request.full_url:
            return _Response({"status": "unloaded"})
        raise _strata_context_error(module, request, 8192, available, status=status)

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=True, config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError):
        _enhance_strata(client, max_tokens=8192)
    assert len(requests) == 2
    assert requests[-1].endswith("/v1/unload?wait=1")


@pytest.mark.parametrize("explicit_limit", [None, 8192])
@pytest.mark.parametrize("message", [
    "request exceeds the context window: no output space remains",
    "Invalid temperature; max_tokens (at most 2048 here)",
])
def test_strata_does_not_guess_an_output_limit_or_retry_other_errors(tmp_path, explicit_limit, message):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request.full_url)
        raise module.urllib.error.HTTPError(
            request.full_url, 400, "failed", {},
            io.BytesIO(json.dumps({"error": {"message": message}}).encode()),
        )

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=False,
        config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError):
        _enhance_strata(client, max_tokens=explicit_limit)
    assert len(requests) == 1


@pytest.mark.parametrize("retry_failure", ["context", "timeout"])
def test_strata_context_fit_retries_once_and_unloads_after_retry_failure(tmp_path, retry_failure):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request.full_url)
        if "/unload" in request.full_url:
            return _Response({"status": "unloaded"})
        if len(requests) == 1:
            raise _strata_context_error(module, request, 8192, 2048)
        if retry_failure == "timeout":
            raise TimeoutError("retry timed out")
        raise _strata_context_error(module, request, 2048, 1024)

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=True, config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError, match="timed out" if retry_failure == "timeout" else "at most 1024"):
        _enhance_strata(client, max_tokens=8192)
    assert len(requests) == 3
    assert requests[-1].endswith("/v1/unload?wait=1")


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
def test_strata_thinking_setting_returns_final_answer_only(tmp_path, effort):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(json.loads(request.data))
        return _Response({"choices": [{"message": {
            "content": "Final video prompt", "reasoning_content": "Private thinking",
            "thinking": "Other thinking", "reasoning": "More thinking",
        }}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=False, reasoning_effort=effort,
        config_path=tmp_path / "missing.yaml", opener=opener,
    )
    assert _enhance_strata(client).prompt == "Final video prompt"
    assert requests[0]["reasoning_effort"] == effort
    assert "chat_template_kwargs" not in requests[0]
    assert set(requests[0]).isdisjoint({
        "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
        "frequency_penalty", "penalty_last_n", "reasoning_budget_tokens",
    })


@pytest.mark.parametrize("content", [
    "<think>Private thinking</think>Final video prompt",
    "<THINK>Private\nthinking</THINK>```text\nFinal video prompt\n```",
    "Private thinking</think>Final video prompt",
    "<thinking>Private thinking</thinking>Final video prompt",
    "<think>Private <think>nested thinking</think> more thinking</think>Final video prompt",
    "Final <think>Private thinking</think>video prompt",
    [
        {"type": "thinking", "text": "Private thinking"},
        {"type": "reasoning", "text": "More private thinking"},
        {"type": "text", "text": "Final video prompt"},
    ],
    [{"type": "output_text", "text": "Final video prompt"}],
])
def test_strata_filters_thinking_from_answer_content(tmp_path, content):
    module = _load_module()
    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=False,
        config_path=tmp_path / "missing.yaml",
        opener=lambda request, timeout: _Response({"choices": [{"message": {"content": content}}]}),
    )
    assert _enhance_strata(client).prompt == "Final video prompt"


@pytest.mark.parametrize("content", [
    None, "<think>Private thinking</think>", "<think>Unfinished thinking",
    [{"type": "thinking", "text": "Only private thinking"}],
])
def test_strata_rejects_reasoning_only_answers(tmp_path, content):
    module = _load_module()
    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=False,
        config_path=tmp_path / "missing.yaml",
        opener=lambda request, timeout: _Response({"choices": [{"message": {
            "content": content, "reasoning_content": "Do not use as a fallback",
        }}]}),
    )
    with pytest.raises(module.PromptEnhancerApiError, match="empty prompt"):
        _enhance_strata(client)


@pytest.mark.parametrize("failure", ["http", "timeout", "empty", "missing", "invalid_json"])
def test_strata_unloads_even_when_generation_fails(tmp_path, failure):
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request.full_url)
        if "/unload" in request.full_url:
            return _Response({"status": "not loaded"})
        if failure == "http":
            raise module.urllib.error.HTTPError(request.full_url, 500, "failed", {}, io.BytesIO(b'{"error":{"message":"inference failed"}}'))
        if failure == "timeout":
            raise TimeoutError("timed out")
        if failure == "invalid_json":
            response = _Response({})
            response.read = lambda: b"not json"
            return response
        return _Response({} if failure == "missing" else {"choices": [{"message": {"content": None}}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=True, config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError):
        _enhance_strata(client)
    assert len(requests) == 2
    assert requests[-1].endswith("/v1/unload?wait=1")


@pytest.mark.parametrize("generation_failed", [False, True])
def test_strata_reports_unload_failures_and_preserves_generation_error(tmp_path, generation_failed):
    module = _load_module()

    def opener(request, timeout):
        if "/unload" in request.full_url:
            return _Response({"status": "unsupported"})
        if generation_failed:
            raise TimeoutError("original inference timed out")
        return _Response({"choices": [{"message": {"content": "Enhanced"}}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", force_unload=True, config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError, match="force_unload failed") as exc_info:
        _enhance_strata(client)
    if generation_failed:
        assert "original inference timed out" in str(exc_info.value)


@pytest.mark.parametrize("options", [
    {"base_url": "file:///tmp/strata"}, {"base_url": "http://host/v1?key=secret"},
    {"base_url": "http://user:password@host/v1"},
    {"reasoning_effort": "invalid"}, {"temperature": float("nan")},
    {"temperature": 2.1},
    {"timeout": 0},
])
def test_strata_rejects_invalid_options_before_requests(tmp_path, options):
    module = _load_module()
    with pytest.raises(ValueError):
        module.PromptEnhancerClient(module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", config_path=tmp_path / "missing.yaml", **options)


@pytest.mark.parametrize("key_source", ["config", "environment"])
def test_strata_optional_api_key_configuration(tmp_path, monkeypatch, key_source):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    if key_source == "config":
        config_path.write_text("OPENAI_LOCAL_API_KEY: config-key\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_LOCAL_API_KEY", "ignored-key")
    else:
        monkeypatch.setenv("OPENAI_LOCAL_API_KEY", "environment-key")
    client = module.PromptEnhancerClient(module.OPENAI_COMPATIBLE_MODEL, "", model_name="strata", config_path=config_path)
    assert client.api_key == ("config-key" if key_source == "config" else "environment-key")


@pytest.mark.parametrize(("explicit", "configured", "expected_base"), [
    ("", "https://config.test/proxy/v1", "https://config.test/proxy/v1"),
    ("  ", " https://config.test/proxy/v1/chat/completions ", "https://config.test/proxy/v1"),
    (None, "http://config.test:8080", "http://config.test:8080/v1"),
    (" https://node.test/v1 ", "https://config.test/v1", "https://node.test/v1"),
    ("https://node.test/v1", "file:///invalid-config-url", "https://node.test/v1"),
    ("", "", "http://127.0.0.1:8080/v1"),
    ("", "  ", "http://127.0.0.1:8080/v1"),
    (None, None, "http://127.0.0.1:8080/v1"),
])
def test_strata_address_priority_is_node_config_then_local_default(tmp_path, explicit, configured, expected_base):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    if configured is not None:
        config_path.write_text(f"OPENAI_LOCAL_BASE_URL: {json.dumps(configured)}\n", encoding="utf-8")
    requests = []

    def opener(request, timeout):
        requests.append(request.full_url)
        if "/unload" in request.full_url:
            return _Response({"status": "unloaded"})
        return _Response({"choices": [{"message": {"content": "Enhanced"}}]})

    options = {} if explicit is None else {"base_url": explicit}
    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "local-key", model_name="strata", force_unload=True, config_path=config_path, opener=opener, **options,
    )
    assert _enhance_strata(client).prompt == "Enhanced"
    assert requests == [f"{expected_base}/chat/completions", f"{expected_base}/unload?wait=1"]


def test_strata_reports_invalid_configured_address_when_node_address_is_empty(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("OPENAI_LOCAL_BASE_URL: file:///invalid-config-url\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Local base_url must be an HTTP or HTTPS URL"):
        module.PromptEnhancerClient(module.OPENAI_COMPATIBLE_MODEL, "local-key", model_name="strata", config_path=config_path)


@pytest.mark.parametrize(("explicit", "configured", "expected"), [
    ("manual-model", "configured-model", "manual-model"),
    ("  manual-model  ", "configured-model", "manual-model"),
    ("", "configured-model", "configured-model"),
    ("", "", "served-model"),
])
def test_local_model_name_priority_and_default_request_parameters(tmp_path, explicit, configured, expected):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"OPENAI_LOCAL_MODEL_NAME: {json.dumps(configured)}\n", encoding="utf-8")
    requests = []

    def opener(request, timeout):
        requests.append((request.get_method(), request.full_url, json.loads(request.data) if request.data else None))
        if request.get_method() == "GET":
            return _Response({"data": [{"id": "served-model"}]})
        return _Response({"choices": [{"message": {
            "content": "<think>Private reasoning</think>Final prompt", "reasoning_content": "Private reasoning",
        }}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name=explicit,
        config_path=config_path, opener=opener,
    )
    assert _enhance_strata(client).prompt == "Final prompt"
    assert len(requests) == (2 if not explicit.strip() and not configured else 1)
    if len(requests) == 2:
        assert requests[0] == ("GET", "http://127.0.0.1:8080/v1/models", None)
    payload = requests[-1][2]
    assert payload["model"] == expected
    assert payload["reasoning_effort"] == "high"
    assert set(payload).isdisjoint({"max_tokens", "force_unload", "chat_template_kwargs", "enable_thinking"})
    assert not any("/unload" in url for _, url, _ in requests)


def test_local_model_discovery_accepts_strata_aliases_and_uses_proxy_prefix(tmp_path):
    module = _load_module()
    urls = []

    def opener(request, timeout):
        urls.append(request.full_url)
        if request.get_method() == "GET":
            return _Response({"data": [
                {"id": "served-qwen", "aliases": ["strata", "qwen"]},
                {"id": "strata", "aliases": ["strata", "qwen"]},
                {"id": "qwen", "aliases": ["strata", "qwen"]},
            ]})
        assert json.loads(request.data)["model"] == "served-qwen"
        return _Response({"choices": [{"message": {"content": "Final prompt"}}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", base_url="https://host.test/proxy/v1/chat/completions",
        config_path=tmp_path / "missing.yaml", opener=opener,
    )
    assert _enhance_strata(client).prompt == "Final prompt"
    assert urls == ["https://host.test/proxy/v1/models", "https://host.test/proxy/v1/chat/completions"]


@pytest.mark.parametrize("response", [
    {}, {"data": []}, {"data": "invalid"}, {"data": [{"id": " "}]},
    {"data": [{"id": "model-one"}, {"id": "model-two"}]},
])
def test_local_model_discovery_requires_an_unambiguous_model(tmp_path, response):
    module = _load_module()
    methods = []

    def opener(request, timeout):
        methods.append(request.get_method())
        return _Response(response)

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError, match="Fill model_name explicitly"):
        _enhance_strata(client)
    assert methods == ["GET"]


def test_local_model_discovery_failure_requests_an_explicit_model_name(tmp_path):
    module = _load_module()

    def opener(request, timeout):
        raise module.urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"{}"))

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", config_path=tmp_path / "missing.yaml", opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError, match="Fill model_name explicitly.*HTTP 404"):
        _enhance_strata(client)


def test_local_configuration_overrides_legacy_strata_configuration(tmp_path, monkeypatch):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "OPENAI_LOCAL_BASE_URL: https://local.test/v1\n"
        "OPENAI_LOCAL_API_KEY: local-key\n"
        "OPENAI_LOCAL_MODEL_NAME: configured-model\n"
        "STRATA_BASE_URL: https://legacy.test/v1\nSTRATA_API_KEY: legacy-key\n", encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_LOCAL_API_KEY", "environment-key")
    monkeypatch.setenv("STRATA_API_KEY", "legacy-environment-key")
    client = module.PromptEnhancerClient(module.OPENAI_COMPATIBLE_MODEL, "", config_path=config_path)
    assert client.config.endpoint == "https://local.test/v1/chat/completions"
    assert client.config.api_model == "configured-model"
    assert client.api_key == "local-key"


def test_legacy_strata_configuration_is_ignored(tmp_path, monkeypatch):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "STRATA_BASE_URL: https://legacy.test/v1\nSTRATA_API_KEY: legacy-key\n", encoding="utf-8",
    )
    monkeypatch.setenv("STRATA_BASE_URL", "https://legacy.test/v1")
    monkeypatch.setenv("STRATA_API_KEY", "legacy-environment-key")
    client = module.PromptEnhancerClient(module.OPENAI_COMPATIBLE_MODEL, "", config_path=config_path)
    assert client.config.endpoint == f"{module.OPENAI_LOCAL_DEFAULT_BASE_URL}/chat/completions"
    assert client.api_key == ""


def test_legacy_strata_provider_name_remains_accepted_by_client(tmp_path):
    module = _load_module()
    client = module.PromptEnhancerClient(
        module.STRATA_MODEL, "", model_name="strata", config_path=tmp_path / "missing.yaml",
        opener=lambda request, timeout: _Response({"choices": [{"message": {"content": "Final prompt"}}]}),
    )
    assert _enhance_strata(client).prompt == "Final prompt"


def test_local_explicit_output_limit_is_not_capped_by_an_assumed_model_context(tmp_path):
    module = _load_module()
    payloads = []

    def opener(request, timeout):
        payloads.append(json.loads(request.data))
        return _Response({"choices": [{"message": {"content": "Final prompt"}}]})

    client = module.PromptEnhancerClient(
        module.OPENAI_COMPATIBLE_MODEL, "", model_name="large-context-model",
        config_path=tmp_path / "missing.yaml", opener=opener,
    )
    assert _enhance_strata(client, max_tokens=300000).prompt == "Final prompt"
    assert payloads[0]["max_tokens"] == 300000


@pytest.mark.parametrize(
    ("model_name", "config_key"),
    [
        ("MINIMAX_MODEL", "MINIMAX_API_KEY"),
        ("VOLCENGINE_MODEL", "VOLCENGINE_API_KEY"),
        ("ZHIPU_MODEL", "BIGMODEL_API_KEY"),
        ("RUNNINGHUB_DOUBAO_MODEL", "RUNNINGHUB_API_KEY"),
        ("RUNNINGHUB_GLM_MODEL", "RUNNINGHUB_API_KEY"),
    ],
)
def test_client_reads_empty_api_key_from_vendor_config(
    tmp_path,
    monkeypatch,
    model_name,
    config_key,
):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"{config_key}: config-secret\n", encoding="utf-8")
    for environment_name in (
        "MINIMAX_API_KEY",
        "VOLCENGINE_API_KEY",
        "BIGMODEL_API_KEY",
        "RUNNINGHUB_API_KEY",
        "ARK_API_KEY",
        "ZHIPU_API_KEY",
    ):
        monkeypatch.delenv(environment_name, raising=False)

    client = module.PromptEnhancerClient(
        getattr(module, model_name),
        "",
        config_path=config_path,
    )

    assert client.api_key == "config-secret"


def test_explicit_api_key_takes_priority_over_config(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("MINIMAX_API_KEY: config-secret\n", encoding="utf-8")

    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "widget-secret",
        config_path=config_path,
    )

    assert client.api_key == "widget-secret"


def test_minimax_upload_configuration_is_fixed(tmp_path):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "MINIMAX_UPLOAD_ENDPOINT: https://upload.example.test/files",
                "MINIMAX_UPLOAD_PURPOSE: should-not-be-used",
            ]
        ),
        encoding="utf-8",
    )

    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        config_path=config_path,
    )

    assert client.upload_endpoint == "https://api.minimaxi.com/v1/files/upload"
    assert client.upload_purpose == "video_generation_input"


def test_missing_api_key_points_to_vendor_config_key(tmp_path, monkeypatch):
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("VOLCENGINE_API_KEY:\n", encoding="utf-8")
    for environment_name in ("VOLCENGINE_API_KEY", "ARK_API_KEY"):
        monkeypatch.delenv(environment_name, raising=False)

    with pytest.raises(
        ValueError,
        match=(
            "Enter apikey or configure VOLCENGINE_API_KEY "
            "in config.yaml"
        ),
    ):
        module.PromptEnhancerClient(
            module.VOLCENGINE_MODEL,
            "",
            config_path=config_path,
        )


def test_image_tensor_data_uris_flattens_batches():
    module = _load_module()
    images = torch.zeros(2, 4, 5, 3)

    encoded = module.image_tensor_data_uris([images, torch.ones(4, 5, 3)])

    assert len(encoded) == 3
    assert all(value.startswith("data:image/png;base64,") for value in encoded)


def test_image_tensor_data_uris_limits_third_party_images_to_two_megapixels():
    module = _load_module()
    encoded = module.image_tensor_data_uris(
        torch.zeros(1, 1200, 2000, 3),
        max_pixels=2_000_000,
    )

    image_bytes = base64.b64decode(encoded[0].split(",", 1)[1])
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert image.width * image.height <= 2_000_000


def test_video_frame_data_uris_samples_at_most_24_frames():
    module = _load_module()

    class Video:
        def get_components(self):
            return type("Components", (), {"images": torch.zeros(30, 8, 8, 3)})()

    encoded = module.video_frame_data_uris([Video()])

    assert len(encoded) == 24
    assert all(value.startswith("data:image/png;base64,") for value in encoded)


def test_zhipu_preserves_public_video_url_without_sampling_frames():
    module = _load_module()

    class Video:
        def get_stream_source(self):
            return "https://example.com/reference.mp4"

        def get_components(self):
            raise AssertionError("public video URL must not be sampled")

    assert module.prompt_enhancer_video_inputs(module.ZHIPU_MODEL, [Video()]) == [
        "https://example.com/reference.mp4"
    ]


def test_zhipu_samples_local_video_when_only_public_urls_are_supported():
    module = _load_module()

    class Video:
        def get_stream_source(self):
            return "/tmp/reference.mp4"

        def get_components(self):
            return type("Components", (), {"images": torch.zeros(30, 8, 8, 3)})()

    encoded = module.prompt_enhancer_video_inputs(module.ZHIPU_MODEL, [Video()])

    assert len(encoded) == 24
    assert all(value.startswith("data:image/png;base64,") for value in encoded)


def test_volcengine_uses_native_data_uri_for_local_video():
    module = _load_module()

    class Video:
        def get_stream_source(self):
            return b"video-bytes"

        def get_container_format(self):
            return "mp4"

        def get_components(self):
            raise AssertionError("native video data URI must not be sampled")

    encoded = module.prompt_enhancer_video_inputs(module.VOLCENGINE_MODEL, [Video()])

    assert len(encoded) == 1
    assert encoded[0].startswith("data:video/mp4;base64,")


def test_runninghub_uses_native_video_data_uri_with_official_limits(monkeypatch):
    module = _load_module()
    calls = []

    class Video:
        def get_stream_source(self):
            return b"video-bytes"

    monkeypatch.setattr(
        module,
        "video_data_uris",
        lambda values, **kwargs: calls.append(kwargs)
        or ["data:video/mp4;base64,AAAA"],
    )

    encoded = module.prompt_enhancer_video_inputs(
        module.RUNNINGHUB_DOUBAO_MODEL,
        [Video()],
    )

    assert encoded == ["data:video/mp4;base64,AAAA"]
    assert calls == [{"max_bytes": 10 * 1024 * 1024, "max_duration": 15}]


def test_video_data_uri_serializes_active_trim_instead_of_original_stream():
    module = _load_module()

    class Video:
        def get_stream_source(self):
            return b"original-video"

        def get_active_trim_window(self):
            return 2.0, 5.0

        def save_to(self, output_path):
            Path(output_path).write_bytes(b"trimmed-video")

    encoded = module.video_data_uris([Video()])

    assert base64.b64decode(encoded[0].split(",", 1)[1]) == b"trimmed-video"


def test_video_data_uri_enforces_duration_even_when_file_is_under_size_limit(monkeypatch):
    module = _load_module()
    calls = []

    class Video:
        def get_stream_source(self):
            return b"small-long-video"

    monkeypatch.setattr(module, "_probe_video_duration", lambda _source, _data: 20.0)
    monkeypatch.setattr(
        module,
        "_limit_video_for_data_uri",
        lambda source, data, **kwargs: calls.append((source, data, kwargs)) or b"limited",
    )

    encoded = module.video_data_uris(
        [Video()],
        max_bytes=10 * 1024 * 1024,
        max_duration=15,
    )

    assert calls == [(
        b"small-long-video",
        b"small-long-video",
        {"max_bytes": 10 * 1024 * 1024, "max_duration": 15},
    )]
    assert base64.b64decode(encoded[0].split(",", 1)[1]) == b"limited"


@pytest.mark.parametrize(
    ("task_type", "expected_mode", "image_count", "video_count", "audio_count"),
    [
        ("t2v", "t2va", 0, 0, 0),
        ("i2v", "i2va", 2, 0, 0),
        ("l2v", "i2va", 1, 0, 0),
        ("v2v", "r2va", 2, 1, 1),
        ("r2v", "r2va", 2, 1, 1),
        ("vi2v", "r2va", 2, 1, 1),
        ("rv2v", "r2va", 2, 1, 1),
    ],
)
def test_h3_task_type_selects_request_mode_and_media(
    task_type,
    expected_mode,
    image_count,
    video_count,
    audio_count,
):
    module = _load_module()

    mode, images, videos, audios = module.PromptEnhancerClient._h3_media_for_task_type(
        task_type,
        ["image-1", "image-2"],
        ["video-1"],
        ["audio-1"],
    )

    assert mode == expected_mode
    assert len(images) == image_count
    assert len(videos) == video_count
    assert len(audios) == audio_count


def test_h3_i2va_roles_follow_task_type_and_image_count():
    module = _load_module()

    i2v_content = module.PromptEnhancerClient._h3_content(
        "Prompt", "i2v", ["first", "last"], [], []
    )
    l2v_content = module.PromptEnhancerClient._h3_content(
        "Prompt", "l2v", ["last"], [], []
    )

    assert [item.get("role") for item in i2v_content[1:]] == [
        "first_frame",
        "last_frame",
    ]
    assert l2v_content[1]["role"] == "last_frame"


def test_h3_r2va_assigns_roles_by_media_type():
    module = _load_module()

    content = module.PromptEnhancerClient._h3_content(
        "Prompt",
        "rv2v",
        ["image"],
        ["video"],
        ["audio"],
    )

    assert [item.get("role") for item in content[1:]] == [
        "reference_image",
        "reference_video",
        "reference_audio",
    ]


def test_h3_rejects_image_dimensions_before_upload():
    module = _load_module()
    client = module.PromptEnhancerClient(module.MINIMAX_MODEL, "secret")

    with pytest.raises(ValueError, match="between 256 and 5760"):
        client._upload_h3_media(_png_data_uri(128, 256), "image", 1, None)


def test_minimax_async_request_returns_task_id_and_maps_ratio():
    module = _load_module()
    requests = []
    log_calls = []

    def opener(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/v1/files/upload"):
            return _Response({"file": {"file_id": 101}})
        return _Response({"task_id": "task-123"})

    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=opener,
    )
    result = client.enhance(
        system_prompt="System instructions that H3 must ignore. " * 300,
        user_prompt="A dancer turns.",
        task_type="r2v",
        duration=2,
        ratio="9:19",
        seed=7,
        image_urls=[_png_data_uri()],
        return_async=True,
        file_count=2,
        request_logger=lambda node_name, message=None: log_calls.append(
            (node_name, message)
        ),
    )

    assert result.prompt == ""
    assert result.task_id == "task-123"
    assert result.file_ids == "101"
    upload_request, upload_timeout = requests[0]
    assert upload_timeout == 300.0
    assert upload_request.full_url == "https://api.minimaxi.com/v1/files/upload"
    assert b"video_generation_input" in upload_request.data
    request, timeout = requests[1]
    assert timeout == 300.0
    assert request.full_url == "https://api.minimaxi.com/v2/h3_context_ir"
    payload = json.loads(request.data)
    assert payload["duration"] == 4
    assert payload["ratio"] == "9:16"
    assert payload["content"][0] == {
        "type": "text",
        "text": "A dancer turns.",
    }
    assert payload["content"][1]["role"] == "reference_image"
    assert payload["content"][1]["image_url"]["url"] == "mm_file://101"
    assert request.headers["Authorization"] == "Bearer secret"
    messages = [message for _node_name, message in log_calls]
    assert messages[0].startswith("H3 uploading image #1")
    assert messages[1] == "H3 upload image #1 succeeded: file_id=101"
    assert "images=1, videos=0, audios=0, files=2" in messages[2]
    assert messages[3] == "H3 create request succeeded: task_id=task-123"


def test_minimax_rejects_only_user_prompt_over_7000_characters():
    module = _load_module()
    requests = []
    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=lambda request, timeout: requests.append(request),
    )

    with pytest.raises(ValueError, match="cannot exceed 7000 characters"):
        client.enhance(
            system_prompt="Ignored system instructions.",
            user_prompt="x" * 7001,
            task_type="t2v",
            duration=5,
            ratio="16:9",
            seed=0,
        )

    assert requests == []


def test_minimax_returns_uploaded_file_ids_in_media_order():
    module = _load_module()
    upload_ids = iter([101, 202, 303])

    def opener(request, timeout):
        if request.full_url.endswith("/v1/files/upload"):
            return _Response({"file": {"file_id": next(upload_ids)}})
        return _Response({"task_id": "task-files"})

    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=opener,
    )
    result = client.enhance(
        system_prompt="",
        user_prompt="Use all references.",
        task_type="r2v",
        duration=5,
        ratio="adaptive",
        seed=0,
        image_urls=[_png_data_uri()],
        video_urls=["data:video/mp4;base64,AAAA"],
        audio_urls=["data:audio/wav;base64,AAAA"],
        return_async=True,
    )

    assert result.file_ids == "101,202,303"


def test_minimax_rejects_empty_user_prompt_before_upload_or_create_request():
    module = _load_module()
    requests = []
    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=lambda request, timeout: requests.append(request),
    )

    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        client.enhance(
            system_prompt="Ignored system instructions.",
            user_prompt="   ",
            task_type="r2v",
            duration=5,
            ratio="adaptive",
            seed=0,
            image_urls=[_png_data_uri()],
        )

    assert requests == []


def test_minimax_sync_request_polls_every_five_seconds():
    module = _load_module()
    payloads = iter(
        [
            {"file": {"file_id": 404}},
            {"task_id": "task-456"},
            {"task": {"status": "queued"}},
            {
                "task": {
                    "status": "succeeded",
                    "content": {"prompt": "```text\nEnhanced.\n```"},
                }
            },
        ]
    )
    sleeps = []
    statuses = []
    log_messages = []

    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=lambda request, timeout: _Response(next(payloads)),
        sleeper=sleeps.append,
    )
    result = client.enhance(
        system_prompt="",
        user_prompt="A city at night.",
        task_type="i2v",
        duration=5,
        ratio="adaptive",
        seed=0,
        image_urls=[_png_data_uri()],
        poll_interval=5.0,
        poll_callback=statuses.append,
        request_logger=lambda _name, message=None: log_messages.append(message),
    )

    assert result.prompt == "Enhanced."
    assert result.task_id == "task-456"
    assert result.file_ids == "404"
    assert sleeps == [5.0, 5.0]
    assert statuses == ["queued", "succeeded"]
    assert log_messages[-4:] == [
        "H3 create request succeeded: task_id=task-456",
        "H3 polling result: task_id=task-456, status=queued",
        "H3 polling result: task_id=task-456, status=succeeded",
        "H3 task succeeded: task_id=task-456",
    ]


def test_minimax_sync_polling_defaults_to_ten_minutes():
    module = _load_module()

    assert module.PromptEnhancerClient.enhance.__kwdefaults__["poll_timeout"] == 600.0


def test_minimax_sync_polling_stops_at_overall_timeout():
    module = _load_module()
    payloads = iter(
        [
            {"task_id": "task-timeout"},
            {"task": {"status": "queued"}},
        ]
    )
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=lambda request, timeout: _Response(next(payloads)),
        sleeper=sleep,
        clock=lambda: now[0],
    )

    with pytest.raises(
        module.PromptEnhancerApiError,
        match="timed out after 6s: task-timeout",
    ):
        client.enhance(
            system_prompt="",
            user_prompt="Prompt",
            task_type="t2v",
            duration=5,
            ratio="16:9",
            seed=0,
            poll_interval=5,
            poll_timeout=6,
        )


def test_openai_compatible_request_extracts_prompt_without_task_id():
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return _Response({
            "choices": [{"message": {"content": "  ```text\nBetter prompt.\n```  "}}]
        })

    client = module.PromptEnhancerClient(
        module.ZHIPU_MODEL,
        "secret",
        opener=opener,
    )
    result = client.enhance(
        system_prompt="You enhance prompts.",
        user_prompt="A dog runs.",
        task_type="i2v",
        duration=6,
        ratio="16:9",
        seed=42,
        image_urls=["data:image/png;base64,AAAA"],
        return_async=True,
    )

    assert result.prompt == "Better prompt."
    assert result.task_id == ""
    payload = json.loads(requests[0].data)
    assert payload["seed"] == 42
    assert payload["messages"][0] == {
        "role": "system",
        "content": "You enhance prompts.",
    }
    assert payload["messages"][1]["content"][1]["type"] == "image_url"


def test_openai_compatible_request_maps_comfyui_seed_to_int32():
    module = _load_module()
    requests = []

    client = module.PromptEnhancerClient(
        module.ZHIPU_MODEL,
        "secret",
        opener=lambda request, timeout: (
            requests.append(request)
            or _Response({"choices": [{"message": {"content": "Enhanced"}}]})
        ),
    )
    client.enhance(
        system_prompt="",
        user_prompt="Prompt",
        task_type="t2v",
        duration=5,
        ratio="16:9",
        seed=715809353714956,
    )

    payload = json.loads(requests[0].data)
    assert payload["seed"] == 1514229004
    assert 0 <= payload["seed"] <= 0x7FFFFFFF


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("```text\nA cinematic prompt.\n```", "A cinematic prompt."),
        ("  ```text\r\nA cinematic prompt.\r\n```  ", "A cinematic prompt."),
        ("```json\n{\"prompt\": \"keep\"}\n```", "```json\n{\"prompt\": \"keep\"}\n```"),
        ("```text\nMissing closing fence", "```text\nMissing closing fence"),
        ("Plain prompt", "Plain prompt"),
    ],
)
def test_strip_text_code_fence_only_removes_complete_text_fence(prompt, expected):
    module = _load_module()

    assert module.strip_text_code_fence(prompt) == expected


def test_minimax_failed_task_is_logged():
    module = _load_module()
    payloads = iter(
        [
            {"task_id": "task-failed"},
            {"task": {"status": "failed", "error": {"message": "bad media"}}},
        ]
    )
    log_messages = []
    client = module.PromptEnhancerClient(
        module.MINIMAX_MODEL,
        "secret",
        opener=lambda request, timeout: _Response(next(payloads)),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(module.PromptEnhancerApiError, match="bad media"):
        client.enhance(
            system_prompt="",
            user_prompt="Prompt",
            task_type="t2v",
            duration=5,
            ratio="16:9",
            seed=0,
            request_logger=lambda _name, message=None: log_messages.append(message),
        )

    assert log_messages[-2:] == [
        "H3 polling result: task_id=task-failed, status=failed",
        "H3 task failed: task_id=task-failed, error=bad media",
    ]


def test_task_type_controls_openai_compatible_media_parts():
    module = _load_module()
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return _Response({"choices": [{"message": {"content": "Enhanced"}}]})

    client = module.PromptEnhancerClient(module.ZHIPU_MODEL, "secret", opener=opener)
    client.enhance(
        system_prompt="",
        user_prompt="Text only",
        task_type="t2v",
        duration=5,
        ratio="16:9",
        seed=0,
        image_urls=["data:image/png;base64,AAAA"],
        video_urls=["data:video/mp4;base64,AAAA"],
        audio_urls=["data:audio/wav;base64,AAAA"],
    )

    payload = json.loads(requests[0].data)
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text"]
    assert payload["max_tokens"] == 65536


def test_third_party_max_tokens_is_capped_by_model_limit():
    module = _load_module()
    requests = []
    client = module.PromptEnhancerClient(
        module.VOLCENGINE_MODEL,
        "secret",
        opener=lambda request, timeout: (
            requests.append(request)
            or _Response({"choices": [{"message": {"content": "Enhanced"}}]})
        ),
    )

    client.enhance(
        system_prompt="",
        user_prompt="Enhance this",
        task_type="t2v",
        duration=5,
        ratio="adaptive",
        seed=0,
        max_tokens=200000,
    )

    assert json.loads(requests[0].data)["max_tokens"] == 131072


def test_native_video_provider_uses_public_video_url_and_omits_audio():
    module = _load_module()
    requests = []
    client = module.PromptEnhancerClient(
        module.ZHIPU_MODEL,
        "secret",
        opener=lambda request, timeout: (
            requests.append(request)
            or _Response({"choices": [{"message": {"content": "Enhanced"}}]})
        ),
    )

    client.enhance(
        system_prompt="",
        user_prompt="Edit the video",
        task_type="v2v",
        duration=5,
        ratio="16:9",
        seed=0,
        video_urls=["https://example.com/reference.mp4"],
        audio_urls=["data:audio/wav;base64,AAAA"],
    )

    payload = json.loads(requests[0].data)
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "video_url"]


def test_sampled_video_frames_remain_image_parts_for_native_provider():
    module = _load_module()
    requests = []
    client = module.PromptEnhancerClient(
        module.ZHIPU_MODEL,
        "secret",
        opener=lambda request, timeout: (
            requests.append(request)
            or _Response({"choices": [{"message": {"content": "Enhanced"}}]})
        ),
    )

    client.enhance(
        system_prompt="",
        user_prompt="Edit the video",
        task_type="v2v",
        duration=5,
        ratio="16:9",
        seed=0,
        video_urls=["data:image/png;base64,AAAA"],
    )

    payload = json.loads(requests[0].data)
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]


def test_runninghub_uses_video_url_and_omits_unsupported_seed():
    module = _load_module()
    requests = []
    client = module.PromptEnhancerClient(
        module.RUNNINGHUB_GLM_MODEL,
        "secret",
        opener=lambda request, timeout: (
            requests.append(request)
            or _Response({"choices": [{"message": {"content": "Enhanced"}}]})
        ),
    )

    client.enhance(
        system_prompt="",
        user_prompt="Edit the video",
        task_type="v2v",
        duration=5,
        ratio="adaptive",
        seed=123,
        video_urls=["data:video/mp4;base64,AAAA"],
    )

    payload = json.loads(requests[0].data)
    assert [part["type"] for part in payload["messages"][0]["content"]] == [
        "text",
        "video_url",
    ]
    assert "seed" not in payload


def test_http_error_is_normalized_with_provider_message():
    module = _load_module()

    def opener(request, timeout):
        raise module.urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"error":{"message":"rate limited"}}'),
        )

    client = module.PromptEnhancerClient(
        module.ZHIPU_MODEL,
        "secret",
        opener=opener,
    )
    with pytest.raises(module.PromptEnhancerApiError, match="HTTP 429: rate limited"):
        client.enhance(
            system_prompt="",
            user_prompt="Prompt",
            task_type="t2v",
            duration=5,
            ratio="16:9",
            seed=0,
        )
