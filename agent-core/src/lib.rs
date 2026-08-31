//! Provider-neutral, embeddable agent runtime.

use async_trait::async_trait;
use futures::{stream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
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

/// Masks configured secret literals (e.g. provider API keys) out of tool
/// output and error text before it is stored in the session or emitted as an
/// event. `Debug` is hand-written so accidentally printing `AgentConfig`
/// cannot leak the very secrets this exists to protect.
#[derive(Clone, Default)]
pub struct SecretRedactor {
    secrets: Vec<String>,
}

impl std::fmt::Debug for SecretRedactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretRedactor({} pattern(s))", self.secrets.len())
    }
}

impl SecretRedactor {
    pub fn new(secrets: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            secrets: secrets
                .into_iter()
                .map(Into::into)
                .filter(|secret| !secret.is_empty())
                .collect(),
        }
    }
    pub fn redact(&self, text: &str) -> String {
        let mut redacted = text.to_owned();
        for secret in &self.secrets {
            redacted = redacted.replace(secret.as_str(), "[REDACTED]");
        }
        redacted
    }
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub max_turns: usize,
    pub max_tool_calls: usize,
    pub timeout: Duration,
    pub workspace_root: std::path::PathBuf,
    pub context_budget: ContextBudget,
    pub redact: SecretRedactor,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 8,
            max_tool_calls: 16,
            timeout: Duration::from_secs(60),
            workspace_root: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            context_budget: ContextBudget::default(),
            redact: SecretRedactor::default(),
        }
    }
}

/// Context is budgeted with a conservative, provider-neutral character estimate.
#[derive(Clone, Debug)]
pub struct ContextBudget {
    pub max_estimated_tokens: usize,
    pub summary_trigger_tokens: usize,
    /// Caps how large the compacted summary itself can grow. Without this,
    /// a long session's summary accumulates forever since it is never
    /// re-summarized; past the cap, the oldest summary entries are dropped.
    pub max_summary_tokens: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_estimated_tokens: 12_000,
            summary_trigger_tokens: 512,
            max_summary_tokens: 3_000,
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

/// Incremental provider output. `Usage` carries a delta to add to a running
/// total, not a cumulative snapshot: providers report it differently (OpenAI
/// sends one final usage chunk; Anthropic splits input/output tokens across
/// `message_start` and `message_delta`), and deltas compose correctly for both.
#[derive(Clone, Debug)]
pub enum ProviderStreamEvent {
    TextDelta(String),
    ToolCallStarted {
        id: ToolCallId,
        name: String,
    },
    ToolCallArgumentsDelta {
        id: ToolCallId,
        partial_json: String,
    },
    ToolCallCompleted {
        id: ToolCallId,
    },
    Usage(TokenUsage),
}

pub type CompletionStream =
    Pin<Box<dyn Stream<Item = Result<ProviderStreamEvent, ProviderError>> + Send>>;

/// Replays an already-complete response as a single-shot stream, for
/// providers that only implement `complete`.
pub fn one_shot_stream(response: CompletionResponse) -> CompletionStream {
    let mut events = Vec::new();
    for block in &response.content {
        match block {
            ContentBlock::Text(text) => events.push(ProviderStreamEvent::TextDelta(text.clone())),
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                events.push(ProviderStreamEvent::ToolCallStarted {
                    id: id.clone(),
                    name: name.clone(),
                });
                events.push(ProviderStreamEvent::ToolCallArgumentsDelta {
                    id: id.clone(),
                    partial_json: arguments.to_string(),
                });
                events.push(ProviderStreamEvent::ToolCallCompleted { id: id.clone() });
            }
        }
    }
    events.push(ProviderStreamEvent::Usage(response.usage));
    Box::pin(stream::iter(events.into_iter().map(Ok)))
}

/// A provider failure. `retryable` distinguishes transient problems (rate
/// limits, timeouts, 5xx responses) from permanent ones (bad request, auth
/// failure) so callers like `RetryingProvider` don't retry what can never
/// succeed.
#[derive(Clone, Debug, Error)]
#[error("provider error: {message}")]
pub struct ProviderError {
    pub message: String,
    pub status: Option<u16>,
    pub retryable: bool,
}

impl ProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            retryable: false,
        }
    }
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            retryable: true,
        }
    }
    /// Classifies retryability from an HTTP status: 429 and 5xx are transient.
    pub fn from_status(status: u16, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: Some(status),
            retryable: status == 429 || status >= 500,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProviderMetadata {
    pub model: String,
    pub context_window_tokens: Option<usize>,
    pub supports_tools: bool,
    pub supports_streaming: bool,
}

impl Default for ProviderMetadata {
    fn default() -> Self {
        Self {
            model: "unknown".into(),
            context_window_tokens: None,
            supports_tools: true,
            supports_streaming: false,
        }
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata::default()
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;
    /// Providers that can stream incrementally should override this; the
    /// default falls back to `complete` and replays the result as one shot.
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        Ok(one_shot_stream(self.complete(request).await?))
    }
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
                    .ok_or_else(|| ProviderError::new("provider is required"))?,
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
        self.run_bounded(session, cancellation).await
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
        self.run_bounded(session, cancellation).await
    }
    /// Applies the run timeout and persists the session whether or not the run
    /// succeeded, so accumulated work survives failures and can be resumed.
    async fn run_bounded(
        &self,
        mut session: SessionState,
        cancellation: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        tracing::info!(session_id = %session.id, "agent run starting");
        let result = match tokio::time::timeout(
            self.config.timeout,
            self.run_session_inner(&mut session, cancellation),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(AgentError::TimedOut),
        };
        if let Err(error) = &result {
            tracing::warn!(session_id = %session.id, error = %error, "agent run failed");
            self.emit(AgentEvent::Failed {
                error: self.config.redact.redact(&error.to_string()),
            });
            // An interrupted turn may leave tool calls without results, which
            // strict providers reject on resume.
            backfill_unanswered_tool_calls(&mut session);
        }
        if let Some(store) = &self.session_store {
            match store.save(&session).await {
                Ok(()) => self.emit(AgentEvent::SessionSaved {
                    session_id: session.id,
                }),
                // A failed save must not mask an error the run already hit.
                Err(error) if result.is_ok() => return Err(error.into()),
                Err(_) => {}
            }
        }
        result
    }
    async fn run_session_inner(
        &self,
        session: &mut SessionState,
        cancellation: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        let mut usage = TokenUsage::default();
        let mut calls = 0;
        for turn in 1..=self.config.max_turns {
            if cancellation.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
            self.emit(AgentEvent::TurnStarted { turn });
            tracing::debug!(turn, "turn started");
            let context = self.context_for(session);
            let mut stream = tokio::select! {
                _ = cancellation.cancelled() => return Err(AgentError::Cancelled),
                result = self.provider.stream(CompletionRequest {
                    messages: context,
                    tools: self.registry.definitions(),
                }) => result?,
            };

            let mut text = String::new();
            // Preserves the order tool calls first appeared in the stream.
            let mut tool_call_order: Vec<ToolCallId> = Vec::new();
            let mut tool_call_names: HashMap<ToolCallId, String> = HashMap::new();
            let mut tool_call_args: HashMap<ToolCallId, String> = HashMap::new();
            loop {
                let next = tokio::select! {
                    _ = cancellation.cancelled() => return Err(AgentError::Cancelled),
                    next = stream.next() => next,
                };
                let Some(event) = next else { break };
                match event? {
                    ProviderStreamEvent::TextDelta(delta) => {
                        self.emit(AgentEvent::TextDelta(delta.clone()));
                        text.push_str(&delta);
                    }
                    ProviderStreamEvent::ToolCallStarted { id, name } => {
                        if !tool_call_names.contains_key(&id) {
                            tool_call_order.push(id.clone());
                        }
                        tool_call_names.insert(id.clone(), name);
                        tool_call_args.entry(id).or_default();
                    }
                    ProviderStreamEvent::ToolCallArgumentsDelta { id, partial_json } => {
                        tool_call_args
                            .entry(id)
                            .or_default()
                            .push_str(&partial_json);
                    }
                    ProviderStreamEvent::ToolCallCompleted { .. } => {}
                    ProviderStreamEvent::Usage(delta) => {
                        usage.input_tokens += delta.input_tokens;
                        usage.output_tokens += delta.output_tokens;
                    }
                }
            }

            let mut content = Vec::new();
            if !text.is_empty() {
                content.push(ContentBlock::Text(text));
            }
            let mut tool_calls = Vec::new();
            for id in tool_call_order {
                let name = tool_call_names.remove(&id).unwrap_or_default();
                let raw = tool_call_args.remove(&id).unwrap_or_default();
                let arguments = if raw.trim().is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&raw).map_err(|error| {
                        ProviderError::new(format!(
                            "invalid arguments for tool call '{name}': {error}"
                        ))
                    })?
                };
                content.push(ContentBlock::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            let final_answer = content
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text(text) = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            session.messages.push(Message::Assistant { content });

            if tool_calls.is_empty() {
                let result = AgentResult {
                    final_answer,
                    messages: session.messages.clone(),
                    usage,
                    session: session.clone(),
                };
                self.emit(AgentEvent::TurnCompleted {
                    usage: result.usage.clone(),
                });
                self.emit(AgentEvent::Completed {
                    result: result.clone(),
                });
                return Ok(result);
            }

            self.run_tool_calls(session, tool_calls, &cancellation, &mut calls)
                .await?;
            self.emit(AgentEvent::TurnCompleted {
                usage: usage.clone(),
            });
        }
        Err(AgentError::TurnBudgetExceeded)
    }
    /// Resolves permission and approval for each call sequentially (an
    /// approval hook reviews one call at a time), then runs the approved
    /// calls concurrently, and finally records their results in the original
    /// order so the transcript stays deterministic regardless of which
    /// finished first.
    async fn run_tool_calls(
        &self,
        session: &mut SessionState,
        tool_calls: Vec<ToolCall>,
        cancellation: &CancellationToken,
        calls: &mut usize,
    ) -> Result<(), AgentError> {
        let mut to_run: Vec<ToolCall> = Vec::new();
        for mut call in tool_calls {
            *calls += 1;
            if *calls > self.config.max_tool_calls {
                return Err(AgentError::ToolBudgetExceeded);
            }
            self.emit(AgentEvent::ToolCallRequested { call: call.clone() });
            match self.policy.permission(&call.name) {
                // Recoverable problems (denied, unknown, invalid) are fed
                // back to the model instead of ending the run, so it can
                // adjust course.
                ToolPermission::Deny => {
                    let output = ToolOutput {
                        text: format!("Denied: tool '{}' is not permitted by policy", call.name),
                        payload: None,
                    };
                    self.push_tool_result(session, call.id, output);
                    continue;
                }
                ToolPermission::RequireApproval => match self.approval.review(&call).await {
                    Approval::Allow => {}
                    Approval::Modify { arguments } => call.arguments = arguments,
                    Approval::Deny { reason } => {
                        let output = ToolOutput {
                            text: format!("Denied: {reason}"),
                            payload: None,
                        };
                        self.push_tool_result(session, call.id, output);
                        continue;
                    }
                },
                ToolPermission::Allow => {}
            }
            let Some(tool) = self.registry.get(&call.name) else {
                let output = ToolOutput {
                    text: format!("Error: unknown tool '{}'", call.name),
                    payload: None,
                };
                self.push_tool_result(session, call.id, output);
                continue;
            };
            if let Err(reason) = validate_arguments(&tool.parameters(), &call.arguments) {
                let output = ToolOutput {
                    text: format!("Invalid arguments: {reason}"),
                    payload: None,
                };
                self.push_tool_result(session, call.id, output);
                continue;
            }
            to_run.push(call);
        }

        if to_run.is_empty() {
            return Ok(());
        }

        let context = ToolContext {
            session_id: session.id,
            workspace_root: self.config.workspace_root.clone(),
            cancellation: cancellation.clone(),
        };
        let executions: Vec<_> = to_run
            .iter()
            .map(|call| {
                tracing::debug!(
                    tool = call.name.as_str(),
                    call_id = call.id.as_str(),
                    "running tool"
                );
                let tool = self
                    .registry
                    .get(&call.name)
                    .expect("presence validated above");
                tool.run(call.arguments.clone(), &context)
            })
            .collect();
        let outputs = futures::future::join_all(executions).await;

        for (call, outcome) in to_run.into_iter().zip(outputs) {
            let output = match outcome {
                Ok(output) => output,
                Err(ToolError::Cancelled) => return Err(AgentError::Cancelled),
                Err(error) => ToolOutput {
                    text: format!("Error: {error}"),
                    payload: None,
                },
            };
            self.push_tool_result(session, call.id, output);
        }
        Ok(())
    }
    /// Assembles the model context, evicting and summarizing older messages
    /// when the estimated size exceeds the budget. Eviction is persisted to the
    /// session so already-summarized messages are never summarized twice.
    fn context_for(&self, session: &mut SessionState) -> Vec<Message> {
        let budget = &self.config.context_budget;
        let mut evicted_total = 0;
        loop {
            let mut evicted = Vec::new();
            while estimated_context_tokens(session) > budget.max_estimated_tokens {
                let Some(index) = evictable_index(&session.messages) else {
                    break;
                };
                evicted.push(session.messages.remove(index));
            }
            if evicted.is_empty() {
                break;
            }
            evicted_total += evicted.len();
            // Below the trigger the evicted content is small enough to drop.
            if estimated_tokens(&evicted) >= budget.summary_trigger_tokens {
                session.memory.summary = Some(compact_messages(
                    session.memory.summary.as_deref(),
                    &evicted,
                    budget.max_summary_tokens,
                ));
                // The grown summary may put the context back over budget, so
                // loop and evict further if candidates remain.
            }
        }
        if evicted_total > 0 {
            tracing::debug!(evicted = evicted_total, "context compacted");
            self.emit(AgentEvent::ContextCompacted {
                evicted_messages: evicted_total,
            });
        }
        let mut context = session.messages.clone();
        if let Some(summary) = &session.memory.summary {
            context.insert(
                1,
                Message::Summary {
                    content: summary.clone(),
                },
            );
        }
        context
    }
    /// Events are advisory: if the receiver is gone or its buffer is full the
    /// event is dropped rather than stalling the run.
    fn emit(&self, event: AgentEvent) {
        let _ = self.events.try_send(event);
    }
    fn push_tool_result(
        &self,
        session: &mut SessionState,
        call_id: ToolCallId,
        output: ToolOutput,
    ) {
        let output = ToolOutput {
            text: self.config.redact.redact(&output.text),
            payload: output.payload,
        };
        self.emit(AgentEvent::ToolCallFinished {
            call_id: call_id.clone(),
            output: output.clone(),
        });
        session
            .messages
            .push(Message::ToolResult { call_id, output });
    }
}

/// The system prompt (index 0) and the current task's user message are never
/// evicted; tool results go first, then the oldest remaining message.
fn evictable_index(messages: &[Message]) -> Option<usize> {
    let current_user = messages
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }));
    messages
        .iter()
        .position(|message| matches!(message, Message::ToolResult { .. }))
        .or_else(|| {
            messages
                .iter()
                .enumerate()
                .find_map(|(index, _)| (index != 0 && Some(index) != current_user).then_some(index))
        })
}

fn estimated_context_tokens(session: &SessionState) -> usize {
    let summary = session
        .memory
        .summary
        .as_ref()
        .map_or(0, |summary| summary.len().div_ceil(4));
    estimated_tokens(&session.messages) + summary
}

/// Conservative structural check of arguments against the tool's JSON Schema.
/// Only the subset the runtime understands is enforced (object type, required
/// keys, primitive property types); unknown schema constructs are ignored so a
/// valid call is never wrongly rejected.
fn validate_arguments(schema: &Value, args: &Value) -> Result<(), String> {
    let Some(schema) = schema.as_object() else {
        return Ok(());
    };
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Ok(());
    }
    let Some(args) = args.as_object() else {
        return Err("arguments must be a JSON object".into());
    };
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for key in required.iter().filter_map(Value::as_str) {
            if !args.contains_key(key) {
                return Err(format!("missing required argument: {key}"));
            }
        }
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property) in properties {
            let expected = property.get("type").and_then(Value::as_str);
            if let (Some(value), Some(expected)) = (args.get(key), expected) {
                if !json_type_matches(value, expected) {
                    return Err(format!("argument '{key}' must be of type {expected}"));
                }
            }
        }
    }
    Ok(())
}

fn json_type_matches(value: &Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Adds placeholder results for assistant tool calls that never received one,
/// so a session interrupted mid-turn stays resumable.
fn backfill_unanswered_tool_calls(session: &mut SessionState) {
    let answered: HashSet<&ToolCallId> = session
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult { call_id, .. } => Some(call_id),
            _ => None,
        })
        .collect();
    let unanswered: Vec<ToolCallId> = session
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { content } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::ToolCall { id, .. } if !answered.contains(id) => Some(id.clone()),
            _ => None,
        })
        .collect();
    for call_id in unanswered {
        session.messages.push(Message::ToolResult {
            call_id,
            output: ToolOutput {
                text: "Interrupted before this tool call finished.".into(),
                payload: None,
            },
        });
    }
}

fn estimated_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|message| serde_json::to_string(message).map_or(0, |value| value.len().div_ceil(4)))
        .sum()
}

/// Folds newly evicted messages into the existing summary, one bullet line
/// per message. The summary is never re-summarized (there is no LLM call
/// here), so it is instead capped by dropping its oldest lines once it grows
/// past `max_tokens` — otherwise a long session's summary would grow forever.
fn compact_messages(existing: Option<&str>, messages: &[Message], max_tokens: usize) -> String {
    let mut lines: Vec<String> = existing
        .map(|summary| summary.lines().map(str::to_owned).collect())
        .unwrap_or_default();
    lines.extend(messages.iter().map(|message| {
        let serialized =
            serde_json::to_string(message).unwrap_or_else(|_| "[unserializable message]".into());
        format!("- {}", serialized.chars().take(400).collect::<String>())
    }));
    while lines.len() > 1 && estimated_line_tokens(&lines) > max_tokens {
        lines.remove(0);
    }
    lines.join("\n")
}

fn estimated_line_tokens(lines: &[String]) -> usize {
    lines.iter().map(|line| line.len().div_ceil(4)).sum()
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
            serde_json::json!({
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}}
            })
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
                ..Default::default()
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

        let context = agent.context_for(&mut session);
        assert!(matches!(context.first(), Some(Message::System { .. })));
        assert!(context.iter().any(
            |message| matches!(message, Message::User { content } if content == &vec![ContentBlock::Text("current task".into())])
        ),
            "context was {context:?}"
        );
        assert!(session.memory.summary.as_deref().unwrap().contains("old"));
    }

    #[tokio::test]
    async fn compaction_never_resummarizes_evicted_messages() {
        let config = AgentConfig {
            context_budget: ContextBudget {
                max_estimated_tokens: 40,
                summary_trigger_tokens: 1,
                ..Default::default()
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
        let mut session = SessionState::new("system".into(), "task".into());
        session.messages.push(Message::ToolResult {
            call_id: "old".into(),
            output: ToolOutput {
                text: "x".repeat(500),
                payload: None,
            },
        });

        agent.context_for(&mut session);
        let summary = session.memory.summary.clone().unwrap();
        let remaining = session.messages.len();
        assert!(!session
            .messages
            .iter()
            .any(|message| matches!(message, Message::ToolResult { .. })));

        // A second assembly finds nothing new to evict: the summary must not
        // grow and no message may be summarized twice.
        agent.context_for(&mut session);
        assert_eq!(session.memory.summary.as_deref(), Some(summary.as_str()));
        assert_eq!(session.messages.len(), remaining);
    }

    #[tokio::test]
    async fn denied_and_unknown_tools_feed_back_instead_of_aborting() {
        let provider = ScriptedProvider {
            responses: std::sync::Mutex::new(vec![
                CompletionResponse {
                    content: vec![
                        ContentBlock::ToolCall {
                            id: "1".into(),
                            name: "blocked".into(),
                            arguments: serde_json::json!({}),
                        },
                        ContentBlock::ToolCall {
                            id: "2".into(),
                            name: "missing".into(),
                            arguments: serde_json::json!({}),
                        },
                    ],
                    usage: TokenUsage::default(),
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text("recovered".into())],
                    usage: TokenUsage::default(),
                },
            ]),
        };
        let mut policy = ToolPolicy::default();
        policy.denylist.insert("blocked".into());
        policy
            .per_tool
            .insert("missing".into(), ToolPermission::Allow);
        let (agent, _) = AgentBuilder::new()
            .provider(provider)
            .tool_policy(policy)
            .build()
            .unwrap();
        let result = agent.run("go").await.unwrap();
        assert_eq!(result.final_answer, "recovered");
        assert!(result.messages.iter().any(|message| matches!(
            message,
            Message::ToolResult { call_id, output } if call_id == "1" && output.text.contains("Denied")
        )));
        assert!(result.messages.iter().any(|message| matches!(
            message,
            Message::ToolResult { call_id, output } if call_id == "2" && output.text.contains("unknown tool")
        )));
    }

    #[tokio::test]
    async fn invalid_arguments_are_rejected_before_the_tool_runs() {
        let provider = ScriptedProvider {
            responses: std::sync::Mutex::new(vec![
                CompletionResponse {
                    content: vec![ContentBlock::ToolCall {
                        id: "1".into(),
                        name: "echo".into(),
                        arguments: serde_json::json!({"value": 42}),
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
        let result = agent.run("go").await.unwrap();
        assert_eq!(result.final_answer, "done");
        assert!(result.messages.iter().any(|message| matches!(
            message,
            Message::ToolResult { output, .. }
                if output.text.contains("Invalid arguments") && output.text.contains("value")
        )));
    }

    struct SlowProvider;
    #[async_trait]
    impl Provider for SlowProvider {
        async fn complete(
            &self,
            _: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            tokio::time::sleep(Duration::from_secs(60)).await;
            unreachable!("the run must time out first")
        }
    }

    #[tokio::test]
    async fn timed_out_runs_still_persist_the_session() {
        let store = InMemorySessionStore::default();
        let (agent, _) = AgentBuilder::new()
            .config(AgentConfig {
                timeout: Duration::from_millis(50),
                ..Default::default()
            })
            .provider(SlowProvider)
            .session_store(store.clone())
            .build()
            .unwrap();
        let error = agent.run("task").await.unwrap_err();
        assert!(matches!(error, AgentError::TimedOut));
        assert_eq!(store.sessions.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn summary_growth_is_capped_across_many_evictions() {
        let config = AgentConfig {
            context_budget: ContextBudget {
                max_estimated_tokens: 40,
                summary_trigger_tokens: 1,
                max_summary_tokens: 50,
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
        let mut session = SessionState::new("system".into(), "task".into());
        // Twenty evictions would grow the summary to ~2000 estimated tokens
        // if it were never capped.
        for i in 0..20 {
            session.messages.push(Message::ToolResult {
                call_id: format!("call-{i}"),
                output: ToolOutput {
                    text: "x".repeat(200),
                    payload: None,
                },
            });
            agent.context_for(&mut session);
        }
        let summary_tokens = session.memory.summary.as_deref().unwrap().len().div_ceil(4);
        assert!(
            summary_tokens < 300,
            "summary grew unbounded: {summary_tokens} estimated tokens"
        );
    }

    struct SleepTool;
    #[async_trait]
    impl Tool for SleepTool {
        fn name(&self) -> &str {
            "sleep"
        }
        fn description(&self) -> &str {
            "sleeps briefly"
        }
        fn parameters(&self) -> Value {
            Value::Null
        }
        async fn run(&self, _: Value, _: &ToolContext) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_millis(60)).await;
            Ok(ToolOutput {
                text: "slept".into(),
                payload: None,
            })
        }
    }

    #[tokio::test]
    async fn independent_tool_calls_in_one_turn_run_concurrently() {
        let provider = ScriptedProvider {
            responses: std::sync::Mutex::new(vec![
                CompletionResponse {
                    content: vec![
                        ContentBlock::ToolCall {
                            id: "a".into(),
                            name: "sleep".into(),
                            arguments: serde_json::json!({}),
                        },
                        ContentBlock::ToolCall {
                            id: "b".into(),
                            name: "sleep".into(),
                            arguments: serde_json::json!({}),
                        },
                    ],
                    usage: TokenUsage::default(),
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text("done".into())],
                    usage: TokenUsage::default(),
                },
            ]),
        };
        let registry = ToolRegistry::new()
            .register(SleepTool, ToolSource::Custom)
            .unwrap();
        let mut policy = ToolPolicy::default();
        policy
            .per_tool
            .insert("sleep".into(), ToolPermission::Allow);
        let (agent, _) = AgentBuilder::new()
            .provider(provider)
            .tools(registry)
            .tool_policy(policy)
            .build()
            .unwrap();
        let started = std::time::Instant::now();
        let result = agent.run("go").await.unwrap();
        // Two 60ms calls run sequentially take >=120ms; concurrently, well under it.
        assert!(
            started.elapsed() < Duration::from_millis(110),
            "tool calls ran sequentially instead of concurrently: {:?}",
            started.elapsed()
        );
        let order: Vec<_> = result
            .messages
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(order, vec!["a".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn tool_output_secrets_are_redacted_before_storage() {
        let provider = ScriptedProvider {
            responses: std::sync::Mutex::new(vec![
                CompletionResponse {
                    content: vec![ContentBlock::ToolCall {
                        id: "1".into(),
                        name: "echo".into(),
                        arguments: serde_json::json!({"value": "token sk-secret123 leaked"}),
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
            .config(AgentConfig {
                redact: SecretRedactor::new(["sk-secret123"]),
                ..Default::default()
            })
            .provider(provider)
            .tools(registry)
            .tool_policy(policy)
            .build()
            .unwrap();
        let result = agent.run("go").await.unwrap();
        assert!(result.messages.iter().any(|message| matches!(
            message,
            Message::ToolResult { output, .. }
                if output.text.contains("[REDACTED]") && !output.text.contains("sk-secret123")
        )));
    }

    struct HangingProvider;
    #[async_trait]
    impl Provider for HangingProvider {
        async fn complete(
            &self,
            _: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn cancellation_interrupts_an_in_flight_provider_call() {
        let (agent, _) = AgentBuilder::new()
            .provider(HangingProvider)
            .build()
            .unwrap();
        let cancellation = CancellationToken::new();
        let token = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            token.cancel();
        });
        let started = std::time::Instant::now();
        let error = agent
            .run_with_cancellation("go", cancellation)
            .await
            .unwrap_err();
        assert!(matches!(error, AgentError::Cancelled));
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "cancellation did not interrupt the in-flight call promptly: {:?}",
            started.elapsed()
        );
    }
}
