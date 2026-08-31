use agent_core::{
    AgentBuilder, AgentProfile, CompletionResponse, ContentBlock, ToolPermission, ToolPolicy,
    ToolRegistry, ToolSource,
};
use agent_llm::MockProvider;
use agent_tools::{ListDir, ReadFile};

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
    let registry = ToolRegistry::new()
        .register(ListDir, ToolSource::Builtin)?
        .register(ReadFile, ToolSource::Builtin)?;
    let mut policy = ToolPolicy::default();
    policy
        .per_tool
        .insert("filesystem.list_dir".into(), ToolPermission::Allow);
    policy
        .per_tool
        .insert("filesystem.read_file".into(), ToolPermission::Allow);
    let (agent, mut events) = AgentBuilder::new()
        .profile(AgentProfile {
            name: "demo-agent".into(),
            role: "A safe, embeddable assistant".into(),
            ..Default::default()
        })
        .provider(MockProvider::scripted(responses))
        .tools(registry)
        .tool_policy(policy)
        .build()?;
    let printer = tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            println!("{event:?}");
        }
    });
    let result = agent.run("Inspect this workspace").await?;
    println!("Final answer: {}", result.final_answer);
    printer.await?;
    Ok(())
}
