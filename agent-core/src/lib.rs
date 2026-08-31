//! Provider-neutral, embeddable agent runtime.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub type ToolCallId = String;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: Vec<ContentBlock>,
    },
    Assistant {
        content: Vec<ContentBlock>,
    },
    ToolResult {
        call_id: ToolCallId,
        output: ToolOutput,
    },
    /// A compacted representation of older session history.
    Summary {
        content: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ContentBlock {
    Text(String),
    ToolCall {
        id: ToolCallId,
        name: String,
        arguments: Value,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolOutput {
    pub text: String,
    pub payload: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolSource {
    Builtin,
    Custom,
    Mcp { server: String },
}

#[derive(Clone, Debug)]
pub struct ToolContext {
    pub session_id: Uuid,
    pub workspace_root: std::path::PathBuf,
    pub cancellation: CancellationToken,
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error("tool execution was cancelled")]
    Cancelled,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn run(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}

struct RegisteredTool {
    tool: Arc<dyn Tool>,
    source: ToolSource,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register<T: Tool + 'static>(
        mut self,
        tool: T,
        source: ToolSource,
    ) -> Result<Self, AgentError> {
        let name = tool.name().to_owned();
        if self.tools.contains_key(&name) {
            return Err(AgentError::DuplicateTool(name));
        }
        self.tools.insert(
            name,
            RegisteredTool {
                tool: Arc::new(tool),
                source,
            },
        );
        Ok(self)
    }
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .get(name)
            .map(|registered| registered.tool.as_ref())
    }
    pub fn source(&self, name: &str) -> Option<&ToolSource> {
        self.tools.get(name).map(|registered| &registered.source)
    }
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|r| ToolDefinition {
                name: r.tool.name().into(),
                description: r.tool.description().into(),
                parameters: r.tool.parameters(),
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentProfile {
    pub name: String,
    pub role: String,
    pub instructions: Vec<String>,
    pub goals: Vec<String>,
}

impl AgentProfile {
    fn system_prompt(&self) -> String {
        let mut lines = vec![format!("You are {}. {}", self.name, self.role)];
        lines.extend(self.instructions.iter().cloned());
        lines.extend(self.goals.iter().map(|goal| format!("Goal: {goal}")));
        lines.join("\n")
    }
}

#[derive(Clone, Debug, Default)]
pub enum ToolPermission {
    Allow,
    Deny,
    #[default]
    RequireApproval,
}

#[derive(Clone, Debug, Default)]
pub struct ToolPolicy {
    pub default: ToolPermission,
    pub per_tool: HashMap<String, ToolPermission>,
    pub denylist: HashSet<String>,
}

impl ToolPolicy {
    pub fn permission(&self, name: &str) -> ToolPermission {
        if self.denylist.contains(name) {
            ToolPermission::Deny
        } else {
            self.per_tool
                .get(name)
                .cloned()
                .unwrap_or_else(|| self.default.clone())
        }
    }
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub max_turns: usize,
    pub max_tool_calls: usize,
    pub timeout: Duration,
    pub workspace_root: std::path::PathBuf,
    pub context_budget: ContextBudget,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 8,
            max_tool_calls: 16,
            timeout: Duration::from_secs(60),
            workspace_root: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            context_budget: ContextBudget::default(),
        }
    }
}

/// Context is budgeted with a conservative, provider-neutral character estimate.
#[derive(Clone, Debug)]
pub struct ContextBudget {
    pub max_estimated_tokens: usize,
    pub summary_trigger_tokens: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_estimated_tokens: 12_000,
            summary_trigger_tokens: 512,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryState {
    pub summary: Option<String>,
    pub workspace: HashMap<String, Value>,
}

/// Serializable state owned by the host. The runtime does not prescribe storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionState {
    pub id: Uuid,
    pub current_task: String,
    pub messages: Vec<Message>,
    pub memory: MemoryState,
}

impl SessionState {
    pub fn new(system_prompt: String, task: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            current_task: task.clone(),
            messages: vec![
                Message::System {
                    content: system_prompt,
                },
                Message::User {
                    content: vec![ContentBlock::Text(task)],
                },
            ],
            memory: MemoryState::default(),
        }
    }
}

#[derive(Debug, Error)]
#[error("session store error: {0}")]
pub struct SessionStoreError(pub String);

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn load(&self, id: Uuid) -> Result<Option<SessionState>, SessionStoreError>;
    async fn save(&self, session: &SessionState) -> Result<(), SessionStoreError>;
}

/// A convenient non-persistent store for tests and simple hosts.
#[derive(Clone, Default)]
pub struct InMemorySessionStore {
    sessions: Arc<std::sync::Mutex<HashMap<Uuid, SessionState>>>,
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn load(&self, id: Uuid) -> Result<Option<SessionState>, SessionStoreError> {
        Ok(self
            .sessions
            .lock()
            .map_err(|_| SessionStoreError("in-memory store lock poisoned".into()))?
            .get(&id)
            .cloned())
    }

    async fn save(&self, session: &SessionState) -> Result<(), SessionStoreError> {
        self.sessions
            .lock()
            .map_err(|_| SessionStoreError("in-memory store lock poisoned".into()))?
            .insert(session.id, session.clone());
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Approval {
    Allow,
    Deny { reason: String },
    Modify { arguments: Value },
}

#[async_trait]
pub trait ApprovalHook: Send + Sync {
    async fn review(&self, call: &ToolCall) -> Approval;
}
pub struct DenyApproval;
#[async_trait]
impl ApprovalHook for DenyApproval {
    async fn review(&self, _: &ToolCall) -> Approval {
        Approval::Deny {
            reason: "no approval hook configured".into(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
}
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub usage: TokenUsage,
}

#[derive(Debug, Error)]
#[error("provider error: {0}")]
pub struct ProviderError(pub String);
#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;
}

#[derive(Clone, Debug)]
pub enum AgentEvent {
    TurnStarted {
        turn: usize,
    },
    TextDelta(String),
    ToolCallRequested {
        call: ToolCall,
    },
    ToolCallFinished {
        call_id: ToolCallId,
        output: ToolOutput,
    },
    TurnCompleted {
        usage: TokenUsage,
    },
    Completed {
        result: AgentResult,
    },
    ContextCompacted {
        evicted_messages: usize,
    },
    SessionSaved {
        session_id: Uuid,
    },
    Failed {
        error: String,
    },
}

#[derive(Clone, Debug)]
pub struct AgentResult {
    pub final_answer: String,
    pub messages: Vec<Message>,
    pub usage: TokenUsage,
    pub session: SessionState,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("duplicate tool name: {0}")]
    DuplicateTool(String),
    #[error("tool not found: {0}")]
    UnknownTool(String),
    #[error("tool {0} is denied by policy")]
    Denied(String),
    #[error("tool call budget exceeded")]
    ToolBudgetExceeded,
    #[error("turn budget exceeded")]
    TurnBudgetExceeded,
    #[error("agent timed out")]
    TimedOut,
    #[error("agent was cancelled")]
    Cancelled,
    #[error("session store is not configured")]
    SessionStoreNotConfigured,
    #[error("session not found: {0}")]
    SessionNotFound(Uuid),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    SessionStore(#[from] SessionStoreError),
}

pub struct AgentBuilder {
    profile: AgentProfile,
    config: AgentConfig,
    policy: ToolPolicy,
    registry: ToolRegistry,
    provider: Option<Arc<dyn Provider>>,
    approval: Arc<dyn ApprovalHook>,
    session_store: Option<Arc<dyn SessionStore>>,
}
impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            profile: AgentProfile::default(),
            config: AgentConfig::default(),
            policy: ToolPolicy::default(),
            registry: ToolRegistry::new(),
            provider: None,
            approval: Arc::new(DenyApproval),
            session_store: None,
        }
    }
    pub fn profile(mut self, profile: AgentProfile) -> Self {
        self.profile = profile;
        self
    }
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }
    pub fn tool_policy(mut self, policy: ToolPolicy) -> Self {
        self.policy = policy;
        self
    }
    pub fn tools(mut self, registry: ToolRegistry) -> Self {
        self.registry = registry;
        self
    }
    pub fn provider<P: Provider + 'static>(mut self, provider: P) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }
    pub fn approval_hook<A: ApprovalHook + 'static>(mut self, approval: A) -> Self {
        self.approval = Arc::new(approval);
        self
    }
    pub fn session_store<S: SessionStore + 'static>(mut self, store: S) -> Self {
        self.session_store = Some(Arc::new(store));
        self
    }
    pub fn build(self) -> Result<(Agent, mpsc::Receiver<AgentEvent>), AgentError> {
        let (events, receiver) = mpsc::channel(128);
        Ok((
            Agent {
                profile: self.profile,
                config: self.config,
                policy: self.policy,
                registry: self.registry,
                provider: self
                    .provider
                    .ok_or_else(|| ProviderError("provider is required".into()))?,
                approval: self.approval,
                events,
                session_store: self.session_store,
            },
            receiver,
        ))
    }
}
impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Agent {
    profile: AgentProfile,
    config: AgentConfig,
    policy: ToolPolicy,
    registry: ToolRegistry,
    provider: Arc<dyn Provider>,
    approval: Arc<dyn ApprovalHook>,
    events: mpsc::Sender<AgentEvent>,
    session_store: Option<Arc<dyn SessionStore>>,
}
impl Agent {
    pub async fn run(&self, task: impl Into<String>) -> Result<AgentResult, AgentError> {
        self.run_with_cancellation(task, CancellationToken::new())
            .await
    }
    pub async fn run_with_cancellation(
        &self,
        task: impl Into<String>,
        cancellation: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        let session = SessionState::new(self.profile.system_prompt(), task.into());
        let future = self.run_session_inner(session, cancellation);
        match tokio::time::timeout(self.config.timeout, future).await {
            Ok(result) => result,
            Err(_) => {
                self.emit(AgentEvent::Failed {
                    error: "agent timed out".into(),
                })
                .await;
                Err(AgentError::TimedOut)
            }
        }
    }
    /// Add a user task to a persisted session and continue the same conversation.
    pub async fn continue_session(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        cancellation: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        let store = self
            .session_store
            .as_ref()
            .ok_or(AgentError::SessionStoreNotConfigured)?;
        let mut session = store
            .load(session_id)
            .await?
            .ok_or(AgentError::SessionNotFound(session_id))?;
        session.current_task = task.into();
        session.messages.push(Message::User {
            content: vec![ContentBlock::Text(session.current_task.clone())],
        });
        self.run_session_inner(session, cancellation).await
    }
    async fn run_session_inner(
        &self,
        mut session: SessionState,
        cancellation: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        let mut usage = TokenUsage::default();
        let mut calls = 0;
        for turn in 1..=self.config.max_turns {
            if cancellation.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
            self.emit(AgentEvent::TurnStarted { turn }).await;
            let context = self.context_for(&mut session).await;
            let response = self
                .provider
                .complete(CompletionRequest {
                    messages: context,
                    tools: self.registry.definitions(),
                })
                .await?;
            usage.input_tokens += response.usage.input_tokens;
            usage.output_tokens += response.usage.output_tokens;
            for block in &response.content {
                if let ContentBlock::Text(text) = block {
                    self.emit(AgentEvent::TextDelta(text.clone())).await;
                }
            }
            let tool_calls: Vec<_> = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    }),
                    _ => None,
                })
                .collect();
            session.messages.push(Message::Assistant {
                content: response.content.clone(),
            });
            if tool_calls.is_empty() {
                let final_answer = response
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text(t) = b {
                            Some(t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let result = AgentResult {
                    final_answer,
                    messages: session.messages.clone(),
                    usage,
                    session,
                };
                if let Some(store) = &self.session_store {
                    store.save(&result.session).await?;
                    self.emit(AgentEvent::SessionSaved {
                        session_id: result.session.id,
                    })
                    .await;
                }
                self.emit(AgentEvent::TurnCompleted {
                    usage: result.usage.clone(),
                })
                .await;
                self.emit(AgentEvent::Completed {
                    result: result.clone(),
                })
                .await;
                return Ok(result);
            }
            for mut call in tool_calls {
                calls += 1;
                if calls > self.config.max_tool_calls {
                    return Err(AgentError::ToolBudgetExceeded);
                }
                self.emit(AgentEvent::ToolCallRequested { call: call.clone() })
                    .await;
                match self.policy.permission(&call.name) {
                    ToolPermission::Deny => return Err(AgentError::Denied(call.name)),
                    ToolPermission::RequireApproval => match self.approval.review(&call).await {
                        Approval::Allow => {}
                        Approval::Modify { arguments } => call.arguments = arguments,
                        Approval::Deny { reason } => {
                            let output = ToolOutput {
                                text: format!("Denied: {reason}"),
                                payload: None,
                            };
                            session.messages.push(Message::ToolResult {
                                call_id: call.id,
                                output,
                            });
                            continue;
                        }
                    },
                    ToolPermission::Allow => {}
                }
                let tool = self
                    .registry
                    .get(&call.name)
                    .ok_or_else(|| AgentError::UnknownTool(call.name.clone()))?;
                let context = ToolContext {
                    session_id: session.id,
                    workspace_root: self.config.workspace_root.clone(),
                    cancellation: cancellation.clone(),
                };
                let output = tool.run(call.arguments, &context).await?;
                self.emit(AgentEvent::ToolCallFinished {
                    call_id: call.id.clone(),
                    output: output.clone(),
                })
                .await;
                session.messages.push(Message::ToolResult {
                    call_id: call.id,
                    output,
                });
            }
            self.emit(AgentEvent::TurnCompleted {
                usage: usage.clone(),
            })
            .await;
        }
        Err(AgentError::TurnBudgetExceeded)
    }
    async fn context_for(&self, session: &mut SessionState) -> Vec<Message> {
        let mut context = session.messages.clone();
        if let Some(summary) = &session.memory.summary {
            context.insert(
                1,
                Message::Summary {
                    content: summary.clone(),
                },
            );
        }
        let mut evicted = Vec::new();
        while estimated_tokens(&context) > self.config.context_budget.max_estimated_tokens {
            let current_user = context
                .iter()
                .rposition(|message| matches!(message, Message::User { .. }));
            let candidate = context
                .iter()
                .position(|message| matches!(message, Message::ToolResult { .. }))
                .or_else(|| {
                    context.iter().enumerate().find_map(|(index, message)| {
                        (index != 0
                            && Some(index) != current_user
                            && !matches!(message, Message::Summary { .. }))
                        .then_some(index)
                    })
                });
            let Some(index) = candidate else { break };
            evicted.push(context.remove(index));
        }
        if !evicted.is_empty() {
            let evicted_tokens = estimated_tokens(&evicted);
            if evicted_tokens >= self.config.context_budget.summary_trigger_tokens {
                session.memory.summary = Some(compact_messages(
                    session.memory.summary.as_deref(),
                    &evicted,
                ));
                context.insert(
                    1,
                    Message::Summary {
                        content: session.memory.summary.clone().unwrap_or_default(),
                    },
                );
                while estimated_tokens(&context) > self.config.context_budget.max_estimated_tokens {
                    let protected_user = context
                        .iter()
                        .rposition(|message| matches!(message, Message::User { .. }));
                    let candidate = context
                        .iter()
                        .position(|message| matches!(message, Message::ToolResult { .. }))
                        .or_else(|| {
                            context.iter().enumerate().find_map(|(index, message)| {
                                (index != 0
                                    && Some(index) != protected_user
                                    && !matches!(message, Message::Summary { .. }))
                                .then_some(index)
                            })
                        });
                    let Some(index) = candidate else { break };
                    context.remove(index);
                }
            }
            self.emit(AgentEvent::ContextCompacted {
                evicted_messages: evicted.len(),
            })
            .await;
        }
        context
    }
    async fn emit(&self, event: AgentEvent) {
        let _ = self.events.send(event).await;
    }
}

fn estimated_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|message| serde_json::to_string(message).map_or(0, |value| value.len().div_ceil(4)))
        .sum()
}

fn compact_messages(existing: Option<&str>, messages: &[Message]) -> String {
    let mut entries = existing.map(str::to_owned).into_iter().collect::<Vec<_>>();
    entries.extend(messages.iter().map(|message| {
        let serialized =
            serde_json::to_string(message).unwrap_or_else(|_| "[unserializable message]".into());
        format!("- {}", serialized.chars().take(400).collect::<String>())
    }));
    entries.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    struct ScriptedProvider {
        responses: std::sync::Mutex<Vec<CompletionResponse>>,
    }
    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn complete(
            &self,
            _: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(self.responses.lock().unwrap().remove(0))
        }
    }
    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes"
        }
        fn parameters(&self) -> Value {
            Value::Null
        }
        async fn run(&self, args: Value, _: &ToolContext) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                text: args["value"].to_string(),
                payload: None,
            })
        }
    }
    #[tokio::test]
    async fn loop_executes_tool_then_returns_answer() {
        let provider = ScriptedProvider {
            responses: std::sync::Mutex::new(vec![
                CompletionResponse {
                    content: vec![ContentBlock::ToolCall {
                        id: "1".into(),
                        name: "echo".into(),
                        arguments: serde_json::json!({"value":"ok"}),
                    }],
                    usage: TokenUsage::default(),
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text("done".into())],
                    usage: TokenUsage::default(),
                },
            ]),
        };
        let registry = ToolRegistry::new()
            .register(Echo, ToolSource::Custom)
            .unwrap();
        let mut policy = ToolPolicy::default();
        policy.per_tool.insert("echo".into(), ToolPermission::Allow);
        let (agent, _) = AgentBuilder::new()
            .provider(provider)
            .tools(registry)
            .tool_policy(policy)
            .build()
            .unwrap();
        assert_eq!(agent.run("go").await.unwrap().final_answer, "done");
    }

    #[tokio::test]
    async fn completed_sessions_are_saved_and_can_be_continued() {
        let provider = ScriptedProvider {
            responses: std::sync::Mutex::new(vec![
                CompletionResponse {
                    content: vec![ContentBlock::Text("first".into())],
                    usage: TokenUsage::default(),
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text("second".into())],
                    usage: TokenUsage::default(),
                },
            ]),
        };
        let store = InMemorySessionStore::default();
        let (agent, _) = AgentBuilder::new()
            .provider(provider)
            .session_store(store.clone())
            .build()
            .unwrap();
        let first = agent.run("start").await.unwrap();
        assert!(store.load(first.session.id).await.unwrap().is_some());
        let resumed = agent
            .continue_session(first.session.id, "follow up", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(resumed.final_answer, "second");
        assert!(resumed
            .messages
            .iter()
            .any(|message| matches!(message, Message::User { content } if content == &vec![ContentBlock::Text("follow up".into())])));
    }

    #[tokio::test]
    async fn compaction_preserves_system_and_current_task() {
        let config = AgentConfig {
            context_budget: ContextBudget {
                max_estimated_tokens: 40,
                summary_trigger_tokens: 1,
            },
            ..Default::default()
        };
        let (agent, _) = AgentBuilder::new()
            .config(config)
            .provider(ScriptedProvider {
                responses: std::sync::Mutex::new(vec![]),
            })
            .build()
            .unwrap();
        let mut session = SessionState::new("system".into(), "old task".into());
        session.messages.push(Message::ToolResult {
            call_id: "old".into(),
            output: ToolOutput {
                text: "x".repeat(500),
                payload: None,
            },
        });
        session.current_task = "current task".into();
        session.messages.push(Message::User {
            content: vec![ContentBlock::Text("current task".into())],
        });

        let context = agent.context_for(&mut session).await;
        assert!(matches!(context.first(), Some(Message::System { .. })));
        assert!(context.iter().any(
            |message| matches!(message, Message::User { content } if content == &vec![ContentBlock::Text("current task".into())])
        ),
            "context was {context:?}"
        );
        assert!(session.memory.summary.as_deref().unwrap().contains("old"));
    }
}
