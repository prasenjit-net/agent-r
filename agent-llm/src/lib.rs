//! Provider implementations. `MockProvider` enables deterministic offline tests.
use agent_core::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, Provider, ProviderError,
    ProviderMetadata, TokenUsage,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
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
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<agent_core::CompletionStream, ProviderError> {
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
            supports_streaming: false,
        }
    }
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(openai_message)
            .collect::<Result<_, _>>()?;
        let tools: Vec<Value> = request.tools.iter().map(|tool| json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.parameters}})).collect();
        let body =
            json!({"model":self.model,"messages":messages,"tools":tools,"tool_choice":"auto"});
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;
        let status = response.status();
        let value: Value = response
            .json()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError(value.to_string()));
        }
        parse_openai_response(value)
    }
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
        .ok_or_else(|| ProviderError("response has no choice".into()))?;
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
                .ok_or_else(|| ProviderError("tool call missing id".into()))?;
            let function = call
                .get("function")
                .ok_or_else(|| ProviderError("tool call missing function".into()))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError("tool call missing name".into()))?;
            let arguments = serde_json::from_str(
                function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}"),
            )
            .map_err(|e| ProviderError(format!("invalid tool arguments: {e}")))?;
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
            .map_err(|_| ProviderError("mock provider lock poisoned".into()))?
            .pop_front()
            .ok_or_else(|| ProviderError("mock provider script exhausted".into()))
    }
}

/// Retries provider failures with exponential backoff.
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
        let mut last_error = ProviderError("retry provider had no attempts".into());
        for attempt in 0..self.max_attempts {
            match self.provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => last_error = error,
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
            return Err(ProviderError(
                "fallback provider requires at least one provider".into(),
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
                Err(error) => errors.push(error.0),
            }
        }
        Err(ProviderError(format!(
            "all fallback providers failed: {}",
            errors.join("; ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
