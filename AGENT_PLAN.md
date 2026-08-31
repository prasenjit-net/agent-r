# General-Purpose Rust Agent Framework Plan

## Goal

Build a lightweight, embeddable Rust agent runtime that can become different kinds of agents depending on the host project, instead of being locked to a single coding-agent role.

The core idea is:

- a small generic agent kernel
- host-defined personality and behavior
- built-in tool packs that can be selected or skipped
- developer-configurable capabilities
- user or project-added tools and MCP servers
- multi-provider LLM support behind a single abstraction

This should be usable as a library in other Rust applications, and it should be flexible enough to become:

- a coding agent
- a research agent
- a support / operations agent
- a workflow automation agent
- a data analysis agent
- a domain-specific internal assistant

---

## 1. Core design principle

The library should not be designed as "a coding agent with optional extras."

It should be designed as:

- a general agent runtime
- plus capability bundles
- plus an execution environment
- plus host-level configuration

That means the runtime remains generic, while the host project chooses which behaviors, tools, and model providers are active.

---

## 2. Product scope

The framework should support:

- a generic agent loop and state machine
- configurable system prompt and behavior profile
- tool registry with built-in, custom, and MCP tools
- multi-provider LLM abstraction
- environment abstraction for file, HTTP, shell, and runtime access
- context memory and session history
- streaming events and host approval hooks
- guardrails and execution limits
- safe extension points for project-specific logic

Explicit non-goals for v1: multi-agent orchestration, vector-database retrieval, and a hosted service layer. These can be built on top of the library later without changing the kernel.

This should remain lightweight and embeddable, with no hard dependency on a specific agent type.

---

## 3. Agent model

An agent should be represented as a composition of several configurable elements:

### Agent profile

```rust
pub struct AgentProfile {
    pub name: String,
    pub role: String,
    pub instructions: Vec<String>,
    pub goals: Vec<String>,
    pub tool_policy: ToolPolicy,
}
```

Note: `instructions` and `goals` are advisory prompt text. Enforceable restrictions (allowlists, denylists, approval requirements, budgets) live in `ToolPolicy` and `AgentConfig`, never in prompt strings. The two must not be conflated: prompt text can be ignored by a model; policy cannot.

### Runtime config

```rust
pub struct AgentConfig {
    pub provider: ProviderConfig,
    pub max_turns: usize,
    pub max_tool_calls: usize,
    pub timeout: Duration,
    pub memory_policy: MemoryPolicy,
    pub context_budget: ContextBudget,
}
```

### Capability bundles

Bundles are typed, not stringly-typed, so misconfiguration fails at compile time:

```rust
pub enum CapabilityBundle {
    Coding,
    Research,
    Web,
    Filesystem,
    Ops,
    Data,
    Knowledge,
}
```

The host project may choose a bundle, combine multiple bundles, or skip them entirely and register tools directly.

---

## 4. Runtime architecture

The runtime should be composed of layers:

1. Agent core
   - orchestrates the conversation loop
   - manages model calls and tool execution
   - enforces budgets and guardrails
   - emits events to the host

2. Provider layer
   - handles model access for OpenAI-compatible, Anthropic, Ollama, Azure, or custom endpoints
   - normalizes each provider's wire format into the internal message model

3. Memory layer
   - stores conversation state, task state, summaries, and long-lived memory

4. Tool layer
   - contains built-in tools, custom tools, and MCP tools behind one trait

5. Environment layer
   - filesystem, HTTP client, process execution, secret access policies, app context

6. Host integration layer
   - builder configuration, event stream, approval hooks

This keeps the library generic while allowing domain-specific agents to be built on top.

---

## 5. Internal message model

This is the hardest part of multi-provider support and must be designed first: providers disagree on how tool calls are represented (Anthropic uses `tool_use` / `tool_result` content blocks; OpenAI-compatible APIs use `tool_calls` on the assistant message and a separate `tool` role), on where the system prompt goes, and on streaming delta formats. The runtime defines one internal model and each provider adapter maps to and from it.

```rust
pub enum Message {
    System { content: String },
    User { content: Vec<ContentBlock> },
    Assistant { content: Vec<ContentBlock> },
    ToolResult { call_id: ToolCallId, output: ToolOutput },
}

pub enum ContentBlock {
    Text(String),
    ToolCall {
        id: ToolCallId,
        name: String,
        arguments: serde_json::Value,
    },
}
```

Requirements:

- lossless round-trip for the constructs the loop depends on (text, tool calls, tool results)
- serializable with `serde` so sessions can be persisted and resumed
- provider adapters own all format quirks; the loop, memory, and tools only ever see this model

Everything downstream — memory, the reasoning loop, event emission — is written against this model, so getting it right up front avoids surgery on every layer later.

---

## 6. Tool abstraction

The `Tool` trait is the most important API in the library. Built-ins, custom tools, and MCP bridging all hang off it.

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the arguments, typically derived via `schemars`.
    fn parameters(&self) -> serde_json::Value;
    async fn run(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}
```

- `ToolContext` gives the tool scoped access to the environment layer (filesystem root, HTTP client, process runner), the session id, and a `CancellationToken`. Tools never reach for global state.
- `ToolOutput` is structured (text plus optional typed payload), so hosts can consume results programmatically, not just as strings.
- Arguments are validated against the schema before `run` is called; schema violations are returned to the model as tool errors, not crashes.
- `async_trait` (or hand-boxed futures) is required for dyn-compatibility even on modern Rust, since `dyn Tool` with a native `async fn` is not object-safe.

### Tool registry

One map, private, with provenance tracked per entry — not separate maps per source, which invites name collisions and special-casing:

```rust
pub enum ToolSource {
    Builtin,
    Custom,
    Mcp { server: String },
}

pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>, // RegisteredTool = Arc<dyn Tool> + ToolSource
}
```

- registration rejects duplicate names; MCP tools may be auto-namespaced (`sales.lookup_customer`) to avoid collisions
- the agent treats all tools uniformly regardless of origin
- a host app can enable a curated built-in subset, add custom business tools, connect MCP servers, or register tools per session

---

## 7. Built-in tool packs

The library ships with common tool packs, but the developer chooses which are compiled in and which are enabled per agent instance. Packs are gated by cargo features on the tools crate (`fs`, `shell`, `http`, `search`, `data`, `workflow`, ...), so unused packs cost nothing.

- filesystem: `list_dir`, `read_file`, `write_file`, `search_files`
- shell / process: `run_command`, `execute_script`, `process_status`
- http / network: `http_get`, `http_post`, `fetch_url`
- search / retrieval: `web_search`, `local_search`, `knowledge_lookup`
- data: `sql_query`, `csv_read`, `json_transform`
- messaging: `send_email`, `send_message`, `webhook_call`
- workflow: `task_queue_submit`, `timer_schedule`, `state_transition`
- developer: `git_status`, `git_diff`, `run_tests`, `compile_project`

The runtime never forces a host app to load all tool packs:

- built-in packs are optional cargo features
- a developer selects a subset via capability bundles or explicit registration
- custom tools can be registered at runtime
- MCP tools are merged into the registry dynamically

---

## 8. Capability profiles

Each profile is just a preset of tool packs and agent instructions, not a different agent engine:

- **Coding**: filesystem, shell, git, test runner, repo search
- **Research**: web search, fetch URL, summarization, document retrieval
- **Ops**: shell, http, workflow state, alert tools, status checks
- **Data**: sql, csv/json readers, analytics tools, report generation
- **Business workflow**: CRM tools, calendar tools, messaging tools, workflow state

---

## 9. Memory and context model

The agent separates context into several categories:

- **session context**: current conversation, user instructions, task state
- **environment context**: repo or workspace, config, credentials policy
- **memory context**: short-term memory, long-term memory, topic summaries
- **domain context**: business objects, workflow state, project metadata

### Memory layers

- `ShortTermMemory`: recent messages and tool results
- `LongTermMemory`: user-specific or workflow-specific summaries
- `WorkspaceMemory`: project structure, key files, recent modifications (used by coding-style profiles; other profiles hold domain objects here)

### Context-window management

This is core-loop behavior, not an optional nicety. The `ContextBudget` policy defines:

- a token budget per model call (derived from the provider's context window minus a reserve for output)
- eviction order when over budget: oldest tool outputs first, then oldest turns, never the system prompt or the current task
- a summarization trigger: when evicted content exceeds a threshold, it is compacted into a summary message rather than silently dropped

### Persistence

Session state (messages, memory, task state) is `serde`-serializable from v1 so hosts can persist and resume sessions. No storage backend is prescribed; the host supplies one behind a small `SessionStore` trait.

### Minimal viable memory

At first, include: the current task, system prompt, recent messages, tool result history, and summarization after several turns. This is enough to be useful without a vector database or heavy retrieval system.

---

## 10. Reasoning loop

The central loop remains generic:

1. gather context and task data
2. assemble messages within the context budget
3. call the selected LLM (streaming)
4. decide if tool use is needed
5. validate tool arguments against the schema
6. run approval hooks for gated tools
7. execute tools (independent calls may run concurrently)
8. feed tool outputs back into context
9. continue until completion or a guardrail limit is hit
10. return the final answer or structured result

The loop works for coding tasks, research tasks, ops workflows, business decision support, and guided automation. The difference is mostly in the tool set and task context, not in the agent core.

Cancellation is first-class: the host holds a `CancellationToken`, and both provider calls and tool executions observe it.

---

## 11. Events, streaming, and approval gates

An embeddable library must let the host observe progress and intercept actions — `run(...).await` returning only a final answer is not enough.

### Event stream

The agent emits events on a channel the host can subscribe to:

```rust
pub enum AgentEvent {
    TurnStarted { turn: usize },
    TextDelta(String),
    ToolCallRequested { call: ToolCall },
    ToolCallFinished { call_id: ToolCallId, output: ToolOutput },
    TurnCompleted { usage: TokenUsage },
    Completed { result: AgentResult },
    Failed { error: AgentError },
}
```

`TextDelta` carries provider streaming output, so hosts can render tokens as they arrive. Hosts that don't care can ignore the stream and just await the result.

### Approval gates

`ToolPolicy` marks tools as `Allow`, `Deny`, or `RequireApproval`. For gated tools the loop calls an async host hook before execution:

```rust
#[async_trait]
pub trait ApprovalHook: Send + Sync {
    async fn review(&self, call: &ToolCall) -> Approval; // Allow | Deny { reason } | Modify { arguments }
}
```

The default hook denies gated calls, so an unconfigured host is safe by default.

---

## 12. Multi-provider support

The runtime is provider-agnostic behind one trait:

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, ProviderError>;
    async fn stream(&self, req: CompletionRequest) -> Result<CompletionStream, ProviderError>;
}
```

### Supported provider families

- OpenAI-compatible APIs
- Anthropic
- Ollama / local models
- Azure OpenAI
- custom self-hosted endpoints

### Design

- a provider registry resolves instances from configuration
- each provider exposes model metadata (context window, capabilities) so the context budget can be derived
- retry with backoff and rate-limit handling live in the provider layer
- fallback lists are supported: primary provider fails, retry with backup, continue with best available

The host can choose one provider per agent, a fallback list, a provider per task class, or a local provider for private tasks and a cloud provider for public ones.

---

## 13. MCP support

MCP is an adapter over the tool abstraction, not a special case in the kernel — and the library must not require MCP for basic use. Because MCP tools implement the same `Tool` trait as everything else, this layer can land after the MVP without touching the core.

### Target design

- `McpClient` connects to an MCP server
- `McpTransport` handles stdio, SSE, or HTTP transport
- `McpToolBridge` converts MCP tool metadata into internal `Tool` objects, namespaced into the registry

### First support target

- stdio-based MCP servers
- JSON-RPC tool discovery
- schema conversion into internal tool definitions

Example registry after bridging: built-in `filesystem.read_file` and `shell.run_command` alongside external `sales.lookup_customer`, `db.query`, `calendar.schedule` — all uniform, all subject to the same `ToolPolicy`.

---

## 14. Safety and governance

Because the same runtime can become many types of agents, safety must be configurable rather than hardcoded — but the enforcement mechanisms are core, not optional.

### Must-have controls

- max turn count and max tool calls per task
- timeout and cancellation policy
- tool allowlist / denylist, including a denylist for dangerous commands
- argument validation before execution
- approval gates for high-risk actions (see section 11)
- secret redaction in logs and tool outputs
- captured stdout/stderr and exit codes for process tools
- sandboxed execution where the platform allows it
- errors surfaced clearly to the host app, never swallowed

### Host-specific risk policy

A coding agent may allow shell execution; a support agent may not. A business workflow agent may permit one set of actions but not others. These policies live in the configuration layer (`ToolPolicy`, `AgentConfig`), not in the runtime kernel — the kernel only enforces them.

---

## 15. Public API

```rust
let (agent, mut events) = AgentBuilder::new()
    .profile(AgentProfile {
        name: "ops-assistant".into(),
        role: "Operational workflow automation".into(),
        instructions: vec![
            "Use tools for confirmation and execution.".into(),
            "Do not perform irreversible actions without approval.".into(),
        ],
        ..Default::default()
    })
    .provider(ProviderConfig::anthropic("claude-sonnet-5"))
    .fallback_provider(ProviderConfig::openai("gpt-4.1"))
    .tools(
        ToolRegistry::new()
            .enable(CapabilityBundle::Filesystem)
            .enable(CapabilityBundle::Ops)
            .register_custom(MyCustomTool)?,
    )
    .approval_hook(MyApprovalHook)
    .build()?;

// Optional: observe progress.
tokio::spawn(async move {
    while let Some(event) = events.recv().await {
        // render deltas, log tool calls, drive a UI...
    }
});

let result = agent.run("Check service health and summarize the incident").await?;
println!("{}", result.final_answer);
```

Same runtime, different role, different tool set. A coding agent host would enable `Coding` instead of `Ops`, point the environment at a repo, and allow shell execution in its `ToolPolicy`.

---

## 16. Crate structure

- `agent-core`
  - message model, agent loop, context, memory, config, events, errors
  - the `Tool`, `Provider`, `ApprovalHook`, and `SessionStore` traits live here so downstream crates depend only on core
- `agent-llm`
  - provider implementations: `openai.rs`, `anthropic.rs`, `ollama.rs`
  - includes `MockProvider` for testing
- `agent-tools`
  - built-in tool packs, one cargo feature per pack (`fs`, `shell`, `http`, `search`, `data`, `workflow`)
- `agent-mcp`
  - `client.rs`, `transport.rs`, `bridge.rs` — optional, post-MVP
- `examples/`
  - two contrasting hosts that prove the general-purpose claim: a coding CLI and an ops assistant

Feature flags — not crate-per-pack — are the mechanism for optional built-ins; the crate split above is by layer, not by domain.

---

## 17. Dependencies

Keep dependencies lean:

- `tokio` for the async runtime, `tokio-util` for `CancellationToken`
- `async-trait` (needed for dyn-compatible async traits even on modern Rust)
- `serde` and `serde_json`
- `schemars` for tool argument schemas
- `thiserror` for the error taxonomy
- `tracing` for internal diagnostics (the event stream, not tracing, is the host-facing surface)
- `reqwest` for HTTP clients
- `uuid` for call and session ids

Avoid over-engineering early; add `time`/`chrono`, storage backends, and retrieval libraries only when a feature actually needs them.

---

## 18. Testing strategy

An agent loop is painful to test against live APIs, so testability is designed in from Phase 1:

- `MockProvider` returns scripted responses, making loop tests deterministic and offline
- recorded-transcript replay: real provider exchanges are captured (with secrets redacted) and replayed as fixtures for regression tests
- tool tests run against a temp-dir environment; the `ToolContext` indirection makes this trivial
- guardrail tests assert that budgets, denylists, and approval gates actually stop execution

---

## 19. Roadmap

### Phase 1: Core runtime
- internal message model
- `Tool` trait and registry
- `Provider` trait plus one implementation and `MockProvider`
- generic agent loop with event channel and cancellation
- deterministic loop tests via mock/replay

### Phase 2: Built-in tools, environment, and guardrails
- environment abstraction (filesystem root, HTTP client, process runner)
- filesystem, shell, and http packs behind feature flags
- `ToolPolicy` enforcement: allowlist, denylist, budgets, approval gates
- structured tool results with captured stdout/stderr and exit codes

### Phase 3: Context and memory
- session state and `SessionStore` persistence/resume
- token budgeting and eviction policy
- summarization of evicted context
- workspace-aware context assembly for the coding profile

### Phase 4: MCP integration
- stdio `McpClient`, JSON-RPC tool discovery
- schema bridge into the internal tool registry
- namespacing and policy application for external tools

### Phase 5: Reliability and breadth
- additional providers, provider fallback
- retry logic and rate limiting
- streaming hardening and richer usage reporting
- remaining tool packs and capability profiles

### First milestone (MVP = Phases 1–2 plus a demo)

- one LLM provider implementation plus the mock
- a registry of basic file and shell tools
- the generic agent loop with events and guardrails
- a command-line example host for demonstration

This is enough to validate the architecture before specializing it for any one domain. MCP, multi-provider fallback, and memory enhancements layer on cleanly afterward because they all sit behind traits defined in Phase 1.

---

## 20. Final recommendation

Build the library as a composable, embeddable core rather than a monolithic app:

- `Agent` as the main runtime
- `AgentBuilder` for host configuration
- `ToolRegistry` for tool loading
- `Provider` registry for model switching
- `Context` and memory for task state
- `Environment` as the execution boundary
- events and approval hooks as the host-facing surface
- `McpClient` as an optional capability layer

Then the host project decides:

- what the agent is for
- which built-in tool packs to enable
- whether to add custom tools
- which MCP servers to connect
- which provider to use
- which safety rules to apply

The key principle:

A general agent kernel plus configurable tool packs and policies is a better foundation than a single-purpose coding agent disguised as a framework.
