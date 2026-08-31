use agent_core::{AgentBuilder, AgentProfile, CompletionResponse, ContentBlock, ToolRegistry};
use agent_llm::MockProvider;
use agent_tools::CapabilityBundle;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let responses = [
        CompletionResponse {
            content: vec![ContentBlock::ToolCall {
                id: "list-1".into(),
                name: "filesystem.list_dir".into(),
                arguments: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            content: vec![ContentBlock::Text(
                "I listed the workspace using the configured filesystem capability.".into(),
            )],
            usage: Default::default(),
        },
    ];
    // The Coding bundle presets a matching tool registry and allow policy;
    // see examples/ops-cli for the same runtime configured as a different
    // kind of agent via CapabilityBundle::Ops.
    let registry = CapabilityBundle::Coding.register(ToolRegistry::new())?;
    let (agent, mut events) = AgentBuilder::new()
        .profile(AgentProfile {
            name: "demo-agent".into(),
            role: "A safe, embeddable assistant".into(),
            ..Default::default()
        })
        .provider(MockProvider::scripted(responses))
        .tools(registry)
        .tool_policy(CapabilityBundle::Coding.allow_policy())
        .build()?;
    let printer = tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            println!("{event:?}");
        }
    });
    let result = agent.run("Inspect this workspace").await?;
    println!("Final answer: {}", result.final_answer);
    // Dropping the agent closes the event channel so the printer can finish.
    drop(agent);
    printer.await?;
    Ok(())
}
