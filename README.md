# agent-r

A lightweight, embeddable Rust agent runtime. It is deliberately provider-neutral and domain-neutral: a host chooses the profile, provider, tools, and safety policy.

The implemented MVP includes:

- a serializable internal message model and generic async agent loop;
- object-safe `Tool`, `Provider`, and `ApprovalHook` extension traits;
- a collision-safe registry with tool provenance;
- turn, tool-call, timeout, cancellation, deny, and approval guardrails;
- optional filesystem and shell tool packs; and
- a deterministic `MockProvider` plus an executable demo; and
- serializable session/memory state, host-provided stores, resumable sessions, and budgeted context compaction.

## Run

```sh
cargo test --workspace
cargo run -p agent-demo
```

See [AGENT_PLAN.md](AGENT_PLAN.md) for the architecture and future phases (MCP, live providers, context compaction, persistence, and broader tool packs).
