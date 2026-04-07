# Spec 02 — LLM Client

> **位置**：`src/maestro/llm/`
> **依赖**：`openai` SDK（阿里云百炼兼容 OpenAI API）、Pydantic v2
> **被依赖**：Planner、Sub-agent、Verifier Tier 3

## 1. 设计原则

- **所有 LLM 调用必须走这一层**，禁止在业务模块里直接 `openai.ChatCompletion.create`
- 统一的 token / cost / latency 统计，每次调用自动累加
- 统一的 structured output 封装（输入 Pydantic model，输出 Pydantic model）
- 统一的 retry、timeout、rate limit 处理
- 统一的日志：每次调用输入输出都有 structured log

## 2. 配置模块

### 2.1 文件：`src/maestro/llm/config.py`

```python
class ModelConfig(BaseModel):
    """Config for one LLM model."""
    name: str                          # 阿里云百炼模型 ID, e.g. "qwen3-max"
    display_name: str                  # 人类可读名，log 里用
    price_input_per_mtok: float        # 美元/百万 input token
    price_output_per_mtok: float       # 美元/百万 output token
    max_tokens_output: int = 4096
    supports_structured_output: bool = True


class ClientConfig(BaseModel):
    """Top-level config."""
    base_url: str                       # 阿里云百炼 endpoint
    api_key: str                        # 从环境变量读取
    models: dict[str, ModelConfig]      # key = role name, e.g. "planner", "subagent", "judge"
    default_timeout_seconds: int = 60
    max_retries: int = 3
    global_semaphore_limit: int = 10    # 全局并发上限，避免 rate limit
```

### 2.2 默认配置（从环境变量和 config file 加载）

默认 `~/.maestro/config.yaml` 或 `MAESTRO_CONFIG` 环境变量指定路径：

```yaml
base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key: "${DASHSCOPE_API_KEY}"  # 从环境变量读取
global_semaphore_limit: 10
models:
  planner:
    name: "qwen3-max"
    display_name: "Qwen3-Max"
    price_input_per_mtok: 2.8   # 示例价格，实际以阿里云计价为准
    price_output_per_mtok: 8.4
    max_tokens_output: 8192
  subagent:
    name: "qwen3-coder-plus"
    display_name: "Qwen3-Coder-Plus"
    price_input_per_mtok: 0.84
    price_output_per_mtok: 3.36
    max_tokens_output: 8192
  judge:
    name: "deepseek-v3"             # 或 "qwen3-coder"
    display_name: "DeepSeek-V3"
    price_input_per_mtok: 0.28
    price_output_per_mtok: 1.12
    max_tokens_output: 2048
```

**重要**：价格数字**不是硬编码**——以配置文件为准，Claude Code 实施时直接使用占位值，真实价格用户运行前自己填。

## 3. Client 主类

### 3.1 文件：`src/maestro/llm/client.py`

```python
T = TypeVar("T", bound=BaseModel)


class LLMCallMetadata(BaseModel):
    """Metadata recorded for each LLM call."""
    model_config = ConfigDict(frozen=True)

    model_name: str
    role: str                           # "planner" / "subagent" / "judge"
    tokens_input: int
    tokens_output: int
    latency_ms: int
    cost_usd: float
    called_at: datetime
    success: bool
    retry_count: int


class LLMClient:
    """Unified LLM client for Maestro."""

    def __init__(self, config: ClientConfig):
        self._config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._semaphore = asyncio.Semaphore(config.global_semaphore_limit)
        self._call_log: list[LLMCallMetadata] = []
        self._log_lock = asyncio.Lock()

    async def call_structured(
        self,
        *,
        role: str,                      # "planner" / "subagent" / "judge"
        messages: list[dict],           # OpenAI chat messages
        output_schema: type[T],         # Pydantic model class
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[T, LLMCallMetadata]:
        """Call LLM with structured output enforcement.

        Returns parsed Pydantic object + call metadata.
        Raises LLMCallError after max_retries exhausted.
        """
        ...

    async def call_text(
        self,
        *,
        role: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, LLMCallMetadata]:
        """Call LLM, return plain text.

        Used mainly for sub-agent when we don't need structured output
        (e.g. debug / fallback). Prefer call_structured when possible.
        """
        ...

    def get_cost_report(self) -> CostReport:
        """Aggregate all calls into cost report."""
        ...
```

### 3.2 `call_structured` 实现要求

**流程**：
1. 获取 semaphore（全局并发控制）
2. 从 config 拿到对应 role 的 model
3. 构造 OpenAI chat.completions 请求，带 `response_format={"type": "json_schema", ...}`
4. 调用 API，记录 latency
5. 解析返回的 JSON 到 Pydantic model
6. 计算 cost = input_tokens × input_price + output_tokens × output_price
7. 生成 `LLMCallMetadata`，append 到 `_call_log`（持锁）
8. 返回

**重试策略**：
- HTTP 5xx 或 429：指数退避重试，最多 `max_retries` 次
- Pydantic 解析失败：允许重试 1 次（给 LLM 一次重新生成机会，prompt 里插一句"Your previous output failed to parse, please strictly follow schema"）
- HTTP 4xx（非 429）：不重试，直接抛 `LLMCallError`
- Timeout：重试，最多 `max_retries` 次

### 3.3 Structured Output 实现

阿里云百炼兼容 OpenAI 的 `response_format={"type": "json_object"}`。对于更严格的 `json_schema`，参考 OpenAI 文档：

```python
response = await self._client.chat.completions.create(
    model=model_config.name,
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": output_schema.__name__,
            "schema": output_schema.model_json_schema(),
            "strict": True,
        },
    },
    temperature=temperature,
    max_tokens=max_tokens or model_config.max_tokens_output,
)
```

**兼容性 fallback**：如果阿里云百炼对 `json_schema` 不完全支持，退化为 `json_object` + 手动 prompt 注入 schema 描述：

```python
# Fallback path
response_format = {"type": "json_object"}
# Prepend system message with schema
schema_str = json.dumps(output_schema.model_json_schema(), indent=2)
messages = [
    {"role": "system", "content": f"Respond strictly as JSON matching this schema:\n{schema_str}"}
] + messages
```

Claude Code 实施时：**先尝试 `json_schema` 路径，如果阿里云百炼不支持，回退到 `json_object` + prompt 注入**。

## 4. 成本统计

### 4.1 CostReport 数据模型

```python
class ModelCostSummary(BaseModel):
    model_name: str
    role: str
    call_count: int
    tokens_input: int
    tokens_output: int
    cost_usd: float
    avg_latency_ms: float


class CostReport(BaseModel):
    """Aggregate cost report from all LLM calls."""
    total_calls: int
    total_tokens_input: int
    total_tokens_output: int
    total_cost_usd: float
    per_model: list[ModelCostSummary]
    per_role: list[ModelCostSummary]

    def to_markdown(self) -> str:
        """Render as markdown table for README / reports."""
        ...
```

### 4.2 Dry-run 模式

`LLMClient` 支持 `dry_run=True` 模式：

```python
class LLMClient:
    def __init__(self, config: ClientConfig, dry_run: bool = False):
        ...
        self._dry_run = dry_run

    async def call_structured(...):
        if self._dry_run:
            # Return a dummy object matching the schema with placeholder fields
            return _make_dummy(output_schema), _dummy_metadata()
        ...
```

Benchmark harness 跑 `--dry-run` 时强制打开此模式，验证 pipeline 不花 API 钱。

## 5. 日志

每次 LLM 调用产生一条 structured log（使用 structlog）：

```python
logger.info(
    "llm_call",
    role=role,
    model=model_config.name,
    tokens_input=tokens_in,
    tokens_output=tokens_out,
    latency_ms=latency,
    cost_usd=cost,
    success=True,
    retry_count=retry,
    # 可选：在 debug mode 下附上 prompt 前 200 字符和 output 前 500 字符
)
```

生产模式默认不记录 prompt / output 内容（避免日志爆炸），debug 模式 `MAESTRO_LOG_LLM_CONTENT=1` 时完整记录。

## 6. 异常

```python
class LLMCallError(Exception):
    """Base exception for LLM call failures."""


class LLMRetryExhausted(LLMCallError):
    """All retries failed."""
    last_error: Exception | None


class LLMOutputParseError(LLMCallError):
    """Output could not be parsed as target schema after retries."""
    raw_output: str
```

## 7. 测试要求

`tests/unit/test_llm_client.py`：
- 使用 `pytest-asyncio` + mock AsyncOpenAI
- 测试 `call_structured` 成功路径
- 测试 Pydantic 解析失败后重试
- 测试 429 错误指数退避
- 测试 semaphore 限流（模拟并发 20 个请求，验证同时 in-flight 不超过 limit）
- 测试 dry-run 模式
- 测试 `CostReport` 聚合正确性

`tests/integration/test_llm_client_live.py`（可选，需设置环境变量 `MAESTRO_LIVE_TEST=1` 才跑）：
- 真实调用阿里云百炼 API，验证 structured output

## 8. 面试 talking points

Claude Code 实施此模块时留下的亮点：

1. **非侵入式成本统计**：业务模块不需要关心 token，统一层自动累加
2. **Semaphore 全局限流**：防止并行爆炸打爆 rate limit
3. **Structured output 双路径**（json_schema + fallback）：体现对不同 provider 兼容性的工程考虑
4. **Dry-run 模式**：体现对"benchmark 烧钱"这个真实工程痛点的理解
