# agent-r

[![CI](https://github.com/prasenjit-net/agent-r/actions/workflows/ci.yml/badge.svg)](https://github.com/prasenjit-net/agent-r/actions/workflows/ci.yml)

A lightweight, embeddable Rust agent runtime. It is deliberately provider-neutral and domain-neutral: a host chooses the profile, provider, tools, and safety policy — the same kernel can become a coding agent, an ops assistant, or something else entirely, without forking the library.

## Crates

| Crate | Description |
| --- | --- |
| [`agent-core`](agent-core) | The runtime: message model, `Tool`/`Provider`/`ApprovalHook`/`SessionStore` traits, the agent loop, context compaction, and guardrails. No I/O and no provider- or tool-specific code. |
| [`agent-llm`](agent-llm) | Provider implementations: OpenAI-compatible (with an Ollama preset) and native Anthropic, plus `RetryingProvider`/`FallbackProvider` reliability wrappers and a deterministic `MockProvider` for tests. |
| [`agent-tools`](agent-tools) | Optional built-in tools (filesystem, shell, HTTP) behind cargo features, plus `CapabilityBundle` presets that pair a tool set with a matching allow policy. |
| [`agent-mcp`](agent-mcp) | An optional stdio JSON-RPC MCP client that bridges remote tools into the same registry as everything else. |
| [`examples/demo`](examples/demo) | A `CapabilityBundle::Coding` agent (filesystem + shell) driven by a scripted `MockProvider`. |
| [`examples/ops-cli`](examples/ops-cli) | The same runtime configured as a different agent via `CapabilityBundle::Ops` (HTTP + shell). |

## Highlights

- a serializable internal message model shared by every provider, so `Assistant`/`ToolResult` turns round-trip whether the model behind it is OpenAI, Anthropic, or a mock;
- real incremental streaming — `CompletionStream` is a live event stream, not a replayed one-shot response — with cancellation racing the in-flight provider call, not just checked between turns;
- turn, tool-call, timeout, cancellation, deny, and approval guardrails enforced in the kernel, not left for each host to reimplement;
- independent tool calls within a turn run concurrently, with results recorded back in a deterministic order;
- resumable sessions: serializable state, a host-provided `SessionStore`, and budgeted context compaction with capped, automatic summarization;
- configurable secret redaction (`AgentConfig::redact`) applied to tool output and failure events before they're stored or emitted;
- `RetryingProvider` only retries errors marked `retryable` (429/5xx, not 4xx); `FallbackProvider` chains providers in order — both preserve real streaming end-to-end instead of degrading to one-shot;
- `CapabilityBundle` presets (`Coding`, `Ops`) pair a tool registry with a matching allow policy, so the same kernel becomes a different agent through configuration, not through a fork.

See [AGENT_PLAN.md](AGENT_PLAN.md) for the full architecture, design rationale, and roadmap.

## Quickstart

```rust
use agent_core::{AgentBuilder, AgentProfile, ToolRegistry};
use agent_llm::OpenAiCompatibleProvider;
use agent_tools::CapabilityBundle;

let registry = CapabilityBundle::Coding.register(ToolRegistry::new())?;
let (agent, mut events) = AgentBuilder::new()
    .profile(AgentProfile {
        name: "coding-agent".into(),
        role: "A careful Rust pair programmer".into(),
        ..Default::default()
    })
    .provider(OpenAiCompatibleProvider::new(api_key, "gpt-4.1"))
    .tools(registry)
    .tool_policy(CapabilityBundle::Coding.allow_policy())
    .build()?;

let result = agent.run("List the files in this workspace").await?;
println!("{}", result.final_answer);
```

`agent_llm::AnthropicProvider` and `agent_llm::OllamaProvider` are drop-in alternatives behind the same `Provider` trait — swapping providers, tool bundles, or policies never touches the agent loop itself.

## Development

```sh
cargo build --workspace --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all -- --check

cargo run -p agent-demo      # Coding bundle, scripted MockProvider
cargo run -p agent-ops-cli   # Ops bundle, scripted MockProvider
```

CI (`.github/workflows/ci.yml`) runs on every push and pull request to `main`: formatting, clippy (`-D warnings`), the test suite across Linux/macOS/Windows, an `agent-tools` feature-combination matrix (each of `fs`/`shell`/`http` built in isolation, since they're independently optional), a docs build, and a `cargo audit` dependency scan. Dependabot keeps both Cargo and Actions dependencies current.

## Status

Early-stage MVP — see [AGENT_PLAN.md](AGENT_PLAN.md) for what's built versus still planned (MSRV policy, remaining tool packs, rate limiting). Not yet published to crates.io.
