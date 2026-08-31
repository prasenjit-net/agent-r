//! Provider implementations: OpenAI-compatible (and Ollama), Anthropic, plus
//! `MockProvider` for deterministic offline tests. `RetryingProvider` and
//! `FallbackProvider` wrap any provider for reliability.
use agent_core::{
    CompletionRequest, CompletionResponse, CompletionStream, ContentBlock, Message, Provider,
    ProviderError, ProviderMetadata, TokenUsage,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::Duration,
};

pub struct MockProvider {
    responses: Mutex<VecDeque<CompletionResponse>>,
}

/// OpenAI Chat Completions-compatible provider. This also supports compatible
/// gateways and local servers by configuring a different base URL.
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

/// Local Ollama preset using its OpenAI-compatible `/v1` endpoint.
pub struct OllamaProvider(OpenAiCompatibleProvider);
impl OllamaProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self(OpenAiCompatibleProvider::with_base_url(
            "http://localhost:11434/v1",
            "ollama",
            model,
        ))
    }
}
#[async_trait]
impl Provider for OllamaProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.0.metadata()
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.0.complete(request).await
    }
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        self.0.stream(request).await
    }
}
impl OpenAiCompatibleProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_base_url("https://api.openai.com/v1", api_key, model)
    }
    pub fn with_base_url(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}
#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            model: self.model.clone(),
            context_window_tokens: None,
            supports_tools: true,
            supports_streaming: true,
        }
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = openai_request_body(&self.model, &request, false)?;
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::new(e.to_string()))?;
        let status = response.status();
        let value: Value = response
            .json()
            .await
            .map_err(|e| ProviderError::new(e.to_string()))?;
        tracing::debug!(model = %self.model, status = status.as_u16(), "openai-compatible completion");
        if !status.is_success() {
            return Err(ProviderError::from_status(
                status.as_u16(),
                value.to_string(),
            ));
        }
        parse_openai_response(value)
    }
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        let body = openai_request_body(&self.model, &request, true)?;
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::new(e.to_string()))?;
        let status = response.status();
        tracing::debug!(model = %self.model, status = status.as_u16(), "openai-compatible stream request");
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status.as_u16(), text));
        }
        let mut tool_call_ids: HashMap<u64, String> = HashMap::new();
        Ok(sse_event_stream(response, move |value| {
            openai_chunk_events(value, &mut tool_call_ids)
        }))
    }
}

fn openai_request_body(
    model: &str,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Value, ProviderError> {
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(openai_message)
        .collect::<Result<_, _>>()?;
    let tools: Vec<Value> = request.tools.iter().map(|tool| json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.parameters}})).collect();
    let mut body =
        json!({"model": model, "messages": messages, "tools": tools, "tool_choice": "auto"});
    if stream {
        body["stream"] = json!(true);
        // Requests a final chunk carrying token usage; without this OpenAI
        // never reports usage on a streamed response.
        body["stream_options"] = json!({"include_usage": true});
    }
    Ok(body)
}

fn openai_message(message: &Message) -> Result<Value, ProviderError> {
    match message {
        Message::System { content } | Message::Summary { content } => {
            Ok(json!({"role":"system","content":content}))
        }
        Message::User { content } => Ok(json!({"role":"user","content":text_blocks(content)})),
        Message::Assistant { content } => {
            let text = text_blocks(content);
            let calls: Vec<Value> = content.iter().filter_map(|block| if let ContentBlock::ToolCall { id,name,arguments } = block { Some(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments.to_string()}})) } else { None }).collect();
            let mut value = json!({"role":"assistant","content":text});
            if !calls.is_empty() {
                value["tool_calls"] = json!(calls);
            }
            Ok(value)
        }
        Message::ToolResult { call_id, output } => {
            Ok(json!({"role":"tool","tool_call_id":call_id,"content":output.text}))
        }
    }
}
fn text_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| {
            if let ContentBlock::Text(text) = block {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}
fn parse_openai_response(value: Value) -> Result<CompletionResponse, ProviderError> {
    let message = value
        .pointer("/choices/0/message")
        .ok_or_else(|| ProviderError::new("response has no choice"))?;
    let mut content = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|s| vec![ContentBlock::Text(s.into())])
        .unwrap_or_default();
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::new("tool call missing id"))?;
            let function = call
                .get("function")
                .ok_or_else(|| ProviderError::new("tool call missing function"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::new("tool call missing name"))?;
            let arguments = serde_json::from_str(
                function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}"),
            )
            .map_err(|e| ProviderError::new(format!("invalid tool arguments: {e}")))?;
            content.push(ContentBlock::ToolCall {
                id: id.into(),
                name: name.into(),
                arguments,
            });
        }
    }
    let usage = TokenUsage {
        input_tokens: value
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: value
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    };
    Ok(CompletionResponse { content, usage })
}

/// Maps one decoded OpenAI streaming chunk to internal events. `tool_call_ids`
/// remembers each tool call's index-to-id mapping, since only the chunk that
/// introduces a tool call carries its id; later argument-delta chunks for the
/// same call only carry the index.
fn openai_chunk_events(
    value: &Value,
    tool_call_ids: &mut HashMap<u64, String>,
) -> Vec<agent_core::ProviderStreamEvent> {
    use agent_core::ProviderStreamEvent;
    let mut events = Vec::new();
    if let Some(usage) = value.get("usage") {
        let input_tokens = usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output_tokens = usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        events.push(ProviderStreamEvent::Usage(TokenUsage {
            input_tokens,
            output_tokens,
        }));
    }
    let Some(delta) = value.pointer("/choices/0/delta") else {
        return events;
    };
    if let Some(text) = delta.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            events.push(ProviderStreamEvent::TextDelta(text.to_string()));
        }
    }
    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                tool_call_ids.insert(index, id.to_string());
                events.push(ProviderStreamEvent::ToolCallStarted {
                    id: id.to_string(),
                    name: name.to_string(),
                });
            }
            if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str) {
                if let Some(id) = tool_call_ids.get(&index) {
                    events.push(ProviderStreamEvent::ToolCallArgumentsDelta {
                        id: id.clone(),
                        partial_json: arguments.to_string(),
                    });
                }
            }
        }
    }
    events
}

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Native Anthropic Messages API provider.
pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.anthropic.com/v1".into(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: 4096,
        }
    }
    /// Anthropic requires `max_tokens` on every request; default is 4096.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            model: self.model.clone(),
            context_window_tokens: None,
            supports_tools: true,
            supports_streaming: true,
        }
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = anthropic_request_body(&self.model, self.max_tokens, &request, false);
        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::new(e.to_string()))?;
        let status = response.status();
        let value: Value = response
            .json()
            .await
            .map_err(|e| ProviderError::new(e.to_string()))?;
        tracing::debug!(model = %self.model, status = status.as_u16(), "anthropic completion");
        if !status.is_success() {
            let message = value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("request failed")
                .to_string();
            return Err(ProviderError::from_status(status.as_u16(), message));
        }
        parse_anthropic_response(value)
    }
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        let body = anthropic_request_body(&self.model, self.max_tokens, &request, true);
        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::new(e.to_string()))?;
        let status = response.status();
        tracing::debug!(model = %self.model, status = status.as_u16(), "anthropic stream request");
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status.as_u16(), text));
        }
        let mut tool_use_ids: HashMap<u64, String> = HashMap::new();
        Ok(sse_event_stream(response, move |value| {
            anthropic_chunk_events(value, &mut tool_use_ids)
        }))
    }
}

/// Builds an Anthropic Messages request body. `System`/`Summary` messages are
/// pulled out into the top-level `system` field (Anthropic has no system
/// role), and consecutive same-role messages are merged, since a `ToolResult`
/// maps to a `user`-role `tool_result` block and several in a row must be
/// batched into one message rather than sent as separate turns.
fn anthropic_request_body(
    model: &str,
    max_tokens: u32,
    request: &CompletionRequest,
    stream: bool,
) -> Value {
    let mut system_parts = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        match message {
            Message::System { content } | Message::Summary { content } => {
                system_parts.push(content.clone());
            }
            Message::User { content } => {
                push_or_merge(&mut messages, "user", anthropic_content_blocks(content));
            }
            Message::Assistant { content } => {
                push_or_merge(
                    &mut messages,
                    "assistant",
                    anthropic_content_blocks(content),
                );
            }
            Message::ToolResult { call_id, output } => {
                push_or_merge(
                    &mut messages,
                    "user",
                    vec![json!({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output.text,
                    })],
                );
            }
        }
    }
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect();
    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
    });
    if !system_parts.is_empty() {
        body["system"] = json!(system_parts.join("\n\n"));
    }
    if stream {
        body["stream"] = json!(true);
    }
    body
}

fn anthropic_content_blocks(content: &[ContentBlock]) -> Vec<Value> {
    content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => json!({"type": "text", "text": text}),
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => json!({"type": "tool_use", "id": id, "name": name, "input": arguments}),
        })
        .collect()
}

fn push_or_merge(messages: &mut Vec<Value>, role: &str, mut blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some(role) {
            if let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) {
                content.append(&mut blocks);
                return;
            }
        }
    }
    messages.push(json!({"role": role, "content": blocks}));
}

fn parse_anthropic_response(value: Value) -> Result<CompletionResponse, ProviderError> {
    let blocks = value
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| ProviderError::new("response has no content"))?;
    let mut content = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                content.push(ContentBlock::Text(text.to_string()));
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ProviderError::new("tool_use block missing id"))?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ProviderError::new("tool_use block missing name"))?;
                let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));
                content.push(ContentBlock::ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments,
                });
            }
            _ => {}
        }
    }
    let usage = TokenUsage {
        input_tokens: value
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: value
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    };
    Ok(CompletionResponse { content, usage })
}

/// Maps one decoded Anthropic streaming event to internal events. Anthropic
/// splits usage across two event types (`message_start` carries input
/// tokens, `message_delta` carries output tokens); both are emitted as
/// separate `Usage` deltas, which the consuming loop adds to a running total.
fn anthropic_chunk_events(
    value: &Value,
    tool_use_ids: &mut HashMap<u64, String>,
) -> Vec<agent_core::ProviderStreamEvent> {
    use agent_core::ProviderStreamEvent;
    match value.get("type").and_then(Value::as_str) {
        Some("content_block_start") => {
            let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
            let Some(block) = value.get("content_block") else {
                return vec![];
            };
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                tool_use_ids.insert(index, id.clone());
                return vec![ProviderStreamEvent::ToolCallStarted { id, name }];
            }
            vec![]
        }
        Some("content_block_delta") => {
            let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
            let Some(delta) = value.get("delta") else {
                return vec![];
            };
            match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => {
                    let text = delta
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    vec![ProviderStreamEvent::TextDelta(text.to_string())]
                }
                Some("input_json_delta") => {
                    let partial = delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    tool_use_ids
                        .get(&index)
                        .map(|id| {
                            vec![ProviderStreamEvent::ToolCallArgumentsDelta {
                                id: id.clone(),
                                partial_json: partial.to_string(),
                            }]
                        })
                        .unwrap_or_default()
                }
                _ => vec![],
            }
        }
        Some("content_block_stop") => {
            let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
            tool_use_ids
                .get(&index)
                .map(|id| vec![ProviderStreamEvent::ToolCallCompleted { id: id.clone() }])
                .unwrap_or_default()
        }
        Some("message_start") => {
            let input_tokens = value
                .pointer("/message/usage/input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            vec![ProviderStreamEvent::Usage(TokenUsage {
                input_tokens,
                output_tokens: 0,
            })]
        }
        Some("message_delta") => {
            let output_tokens = value
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            vec![ProviderStreamEvent::Usage(TokenUsage {
                input_tokens: 0,
                output_tokens,
            })]
        }
        _ => vec![],
    }
}

/// Buffers a chunked HTTP response into SSE events (`data: ...` lines
/// separated by a blank line) and maps each decoded JSON payload to internal
/// stream events via `handle_chunk`, which owns whatever per-stream state it
/// needs (e.g. an index-to-id map for tool calls).
fn sse_event_stream<F>(response: reqwest::Response, mut handle_chunk: F) -> CompletionStream
where
    F: FnMut(&Value) -> Vec<agent_core::ProviderStreamEvent> + Send + 'static,
{
    Box::pin(async_stream::try_stream! {
        let mut bytes = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|e| ProviderError::new(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buffer.find("\n\n") {
                let event_block = buffer[..pos].to_string();
                buffer.drain(..pos + 2);
                for line in event_block.lines() {
                    let Some(data) = line.strip_prefix("data:") else { continue };
                    let data = data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    let value: Value = serde_json::from_str(data)
                        .map_err(|e| ProviderError::new(format!("invalid stream chunk: {e}")))?;
                    for event in handle_chunk(&value) {
                        yield event;
                    }
                }
            }
        }
    })
}

impl MockProvider {
    pub fn scripted(responses: impl IntoIterator<Item = CompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
        }
    }
}
#[async_trait]
impl Provider for MockProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            model: "mock".into(),
            context_window_tokens: Some(usize::MAX),
            supports_tools: true,
            supports_streaming: false,
        }
    }
    async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        self.responses
            .lock()
            .map_err(|_| ProviderError::new("mock provider lock poisoned"))?
            .pop_front()
            .ok_or_else(|| ProviderError::new("mock provider script exhausted"))
    }
}

/// Retries provider failures with exponential backoff. Only errors marked
/// `retryable` (rate limits, timeouts, 5xx) are retried; a permanent failure
/// (bad request, auth) fails immediately instead of wasting attempts.
pub struct RetryingProvider {
    provider: Arc<dyn Provider>,
    max_attempts: usize,
    initial_backoff: Duration,
}
impl RetryingProvider {
    pub fn new<P: Provider + 'static>(
        provider: P,
        max_attempts: usize,
        initial_backoff: Duration,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            max_attempts: max_attempts.max(1),
            initial_backoff,
        }
    }
}
#[async_trait]
impl Provider for RetryingProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.provider.metadata()
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let mut delay = self.initial_backoff;
        let mut last_error = ProviderError::new("retry provider had no attempts");
        for attempt in 0..self.max_attempts {
            match self.provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let retryable = error.retryable;
                    last_error = error;
                    if !retryable {
                        break;
                    }
                }
            }
            if attempt + 1 < self.max_attempts {
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
        }
        Err(last_error)
    }
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        let mut delay = self.initial_backoff;
        let mut last_error = ProviderError::new("retry provider had no attempts");
        for attempt in 0..self.max_attempts {
            match self.provider.stream(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(error) => {
                    let retryable = error.retryable;
                    last_error = error;
                    if !retryable {
                        break;
                    }
                }
            }
            if attempt + 1 < self.max_attempts {
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
        }
        Err(last_error)
    }
}

/// Uses providers in order, allowing a cloud model to fall back to a local one.
pub struct FallbackProvider {
    providers: Vec<Arc<dyn Provider>>,
}
impl FallbackProvider {
    pub fn new(providers: Vec<Arc<dyn Provider>>) -> Result<Self, ProviderError> {
        if providers.is_empty() {
            return Err(ProviderError::new(
                "fallback provider requires at least one provider",
            ));
        }
        Ok(Self { providers })
    }
}
#[async_trait]
impl Provider for FallbackProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.providers[0].metadata()
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let mut errors = Vec::new();
        for provider in &self.providers {
            match provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => errors.push(error.message),
            }
        }
        Err(ProviderError::new(format!(
            "all fallback providers failed: {}",
            errors.join("; ")
        )))
    }
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        let mut errors = Vec::new();
        for provider in &self.providers {
            match provider.stream(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(error) => errors.push(error.message),
            }
        }
        Err(ProviderError::new(format!(
            "all fallback providers failed: {}",
            errors.join("; ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parses_text_tool_calls_and_usage() {
        let response = parse_openai_response(json!({
            "choices": [{"message": {"content": "Checking.", "tool_calls": [{"id":"call_1", "type":"function", "function":{"name":"filesystem.read_file", "arguments":"{\"path\":\"README.md\"}"}}]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}
        })).unwrap();
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.output_tokens, 7);
        assert!(matches!(&response.content[0], ContentBlock::Text(text) if text == "Checking."));
        assert!(
            matches!(&response.content[1], ContentBlock::ToolCall { id, name, arguments } if id == "call_1" && name == "filesystem.read_file" && arguments["path"] == "README.md")
        );
    }

    #[test]
    fn rejects_invalid_tool_arguments() {
        let result = parse_openai_response(
            json!({"choices":[{"message":{"tool_calls":[{"id":"call_1","function":{"name":"tool","arguments":"not json"}}]}}]}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn openai_stream_chunk_maps_text_and_tool_call_deltas() {
        use agent_core::ProviderStreamEvent;
        let mut ids = HashMap::new();
        let started = openai_chunk_events(
            &json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "echo", "arguments": ""}}]}}]}),
            &mut ids,
        );
        assert!(
            matches!(&started[0], ProviderStreamEvent::ToolCallStarted { id, name } if id == "call_1" && name == "echo")
        );

        let args = openai_chunk_events(
            &json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}}]}),
            &mut ids,
        );
        assert!(
            matches!(&args[0], ProviderStreamEvent::ToolCallArgumentsDelta { id, partial_json } if id == "call_1" && partial_json == "{\"a\":1}")
        );

        let text = openai_chunk_events(
            &json!({"choices": [{"delta": {"content": "hi"}}]}),
            &mut ids,
        );
        assert!(matches!(&text[0], ProviderStreamEvent::TextDelta(t) if t == "hi"));

        let usage = openai_chunk_events(
            &json!({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}),
            &mut ids,
        );
        assert!(
            matches!(&usage[0], ProviderStreamEvent::Usage(u) if u.input_tokens == 3 && u.output_tokens == 5)
        );
    }

    #[test]
    fn anthropic_request_merges_consecutive_tool_results_into_one_message() {
        let request = CompletionRequest {
            messages: vec![
                Message::System {
                    content: "sys".into(),
                },
                Message::User {
                    content: vec![ContentBlock::Text("hi".into())],
                },
                Message::Assistant {
                    content: vec![
                        ContentBlock::ToolCall {
                            id: "1".into(),
                            name: "a".into(),
                            arguments: json!({}),
                        },
                        ContentBlock::ToolCall {
                            id: "2".into(),
                            name: "b".into(),
                            arguments: json!({}),
                        },
                    ],
                },
                Message::ToolResult {
                    call_id: "1".into(),
                    output: agent_core::ToolOutput {
                        text: "one".into(),
                        payload: None,
                    },
                },
                Message::ToolResult {
                    call_id: "2".into(),
                    output: agent_core::ToolOutput {
                        text: "two".into(),
                        payload: None,
                    },
                },
            ],
            tools: vec![],
        };
        let body = anthropic_request_body("claude", 4096, &request, false);
        assert_eq!(body["system"], json!("sys"));
        let messages = body["messages"].as_array().unwrap();
        // user, assistant(2 tool_use blocks), user(2 tool_result blocks) merged.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn anthropic_stream_chunk_maps_tool_use_and_split_usage() {
        use agent_core::ProviderStreamEvent;
        let mut ids = HashMap::new();
        let started = anthropic_chunk_events(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "echo"}}),
            &mut ids,
        );
        assert!(
            matches!(&started[0], ProviderStreamEvent::ToolCallStarted { id, name } if id == "toolu_1" && name == "echo")
        );

        let args = anthropic_chunk_events(
            &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":1}"}}),
            &mut ids,
        );
        assert!(
            matches!(&args[0], ProviderStreamEvent::ToolCallArgumentsDelta { id, partial_json } if id == "toolu_1" && partial_json == "{\"a\":1}")
        );

        let start_usage = anthropic_chunk_events(
            &json!({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            &mut ids,
        );
        assert!(
            matches!(&start_usage[0], ProviderStreamEvent::Usage(u) if u.input_tokens == 10 && u.output_tokens == 0)
        );

        let delta_usage = anthropic_chunk_events(
            &json!({"type": "message_delta", "usage": {"output_tokens": 4}}),
            &mut ids,
        );
        assert!(
            matches!(&delta_usage[0], ProviderStreamEvent::Usage(u) if u.input_tokens == 0 && u.output_tokens == 4)
        );
    }

    #[test]
    fn provider_error_from_status_classifies_retryable() {
        assert!(ProviderError::from_status(429, "rate limited").retryable);
        assert!(ProviderError::from_status(500, "server error").retryable);
        assert!(ProviderError::from_status(503, "unavailable").retryable);
        assert!(!ProviderError::from_status(400, "bad request").retryable);
        assert!(!ProviderError::from_status(401, "unauthorized").retryable);
        assert!(!ProviderError::from_status(404, "not found").retryable);
    }

    struct CountingProvider {
        attempts: Arc<AtomicUsize>,
        fail_times: usize,
        retryable: bool,
    }
    #[async_trait]
    impl Provider for CountingProvider {
        async fn complete(
            &self,
            _: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < self.fail_times {
                return if self.retryable {
                    Err(ProviderError::retryable("transient"))
                } else {
                    Err(ProviderError::new("permanent"))
                };
            }
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text("ok".into())],
                usage: TokenUsage::default(),
            })
        }
    }

    fn empty_request() -> CompletionRequest {
        CompletionRequest {
            messages: vec![],
            tools: vec![],
        }
    }

    #[tokio::test]
    async fn retrying_provider_stops_immediately_on_non_retryable_error() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = RetryingProvider::new(
            CountingProvider {
                attempts: attempts.clone(),
                fail_times: usize::MAX,
                retryable: false,
            },
            5,
            Duration::from_millis(1),
        );
        let error = provider.complete(empty_request()).await.unwrap_err();
        assert_eq!(error.message, "permanent");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retrying_provider_retries_transient_errors_until_success() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = RetryingProvider::new(
            CountingProvider {
                attempts: attempts.clone(),
                fail_times: 2,
                retryable: true,
            },
            5,
            Duration::from_millis(1),
        );
        let response = provider.complete(empty_request()).await.unwrap();
        assert!(matches!(&response.content[0], ContentBlock::Text(t) if t == "ok"));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
