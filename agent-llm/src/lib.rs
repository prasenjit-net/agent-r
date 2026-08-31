//! Provider implementations. `MockProvider` enables deterministic offline tests.
use agent_core::{
    CompletionRequest, CompletionResponse, Provider, ProviderError, ProviderMetadata,
};
use async_trait::async_trait;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

pub struct MockProvider {
    responses: Mutex<VecDeque<CompletionResponse>>,
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
