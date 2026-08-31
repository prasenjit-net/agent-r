use agent_core::{AgentBuilder, AgentProfile, CompletionResponse, ContentBlock, ToolRegistry};
use agent_llm::MockProvider;
use agent_tools::CapabilityBundle;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let responses = [
        CompletionResponse {
            content: vec![ContentBlock::ToolCall {
                id: "check-1".into(),
                name: "shell.run_command".into(),
                arguments: serde_json::json!({"program": "echo", "args": ["service healthy"]}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            content: vec![ContentBlock::Text(
                "Service health check passed using the ops capability bundle.".into(),
            )],
            usage: Default::default(),
        },
    ];
    let registry = CapabilityBundle::Ops.register(ToolRegistry::new())?;
    let (agent, mut events) = AgentBuilder::new()
        .profile(AgentProfile {
            name: "ops-assistant".into(),
            role: "Operational health-check automation".into(),
            ..Default::default()
        })
        .provider(MockProvider::scripted(responses))
        .tools(registry)
        .tool_policy(CapabilityBundle::Ops.allow_policy())
        .build()?;
    let printer = tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            println!("{event:?}");
        }
    });
    let result = agent.run("Check service health").await?;
    println!("Final answer: {}", result.final_answer);
    // Dropping the agent closes the event channel so the printer can finish.
    drop(agent);
    printer.await?;
    Ok(())
}
