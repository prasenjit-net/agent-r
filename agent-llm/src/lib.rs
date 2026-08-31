//! Provider implementations. `MockProvider` enables deterministic offline tests.
use agent_core::{CompletionRequest, CompletionResponse, Provider, ProviderError};
use async_trait::async_trait;
use std::{collections::VecDeque, sync::Mutex};

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
    async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        self.responses
            .lock()
            .map_err(|_| ProviderError("mock provider lock poisoned".into()))?
            .pop_front()
            .ok_or_else(|| ProviderError("mock provider script exhausted".into()))
    }
}
