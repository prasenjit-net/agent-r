//! Model Context Protocol support as an adapter over `agent_core::Tool`.
//!
//! This crate currently targets newline-delimited JSON-RPC over stdio, which is
//! the transport used by many local MCP servers. The core runtime has no MCP
//! dependency; callers explicitly discover and bridge only the servers needed.

use std::{collections::HashMap, process::Stdio, sync::Arc};

use agent_core::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSource};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
};

#[derive(Debug, Error)]
pub enum McpError {
    #[error("failed to start MCP server: {0}")]
    Start(#[source] std::io::Error),
    #[error("MCP transport error: {0}")]
    Transport(#[source] std::io::Error),
    #[error("MCP server closed its output stream")]
    Closed,
    #[error("invalid MCP response: {0}")]
    Protocol(String),
    #[error("MCP request failed ({code}): {message}")]
    Remote { code: i64, message: String },
    #[error(transparent)]
    Registry(#[from] agent_core::AgentError),
}

#[derive(Clone, Debug)]
pub struct StdioMcpConfig {
    pub command: String,
    pub args: Vec<String>,
}

impl StdioMcpConfig {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
        }
    }
    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }
}

pub struct McpClient {
    transport: Arc<StdioTransport>,
    next_id: Mutex<u64>,
    /// Replies read off the wire for requests other than the reader's own.
    pending: Mutex<HashMap<u64, JsonRpcResponse>>,
}

struct StdioTransport {
    // Keep the child alive for the lifetime of its streams.
    _child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<Lines<BufReader<ChildStdout>>>,
}

impl McpClient {
    pub async fn connect(config: StdioMcpConfig) -> Result<Arc<Self>, McpError> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(McpError::Start)?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Protocol("server stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Protocol("server stdout unavailable".into()))?;
        let client = Arc::new(Self {
            transport: Arc::new(StdioTransport {
                _child: Mutex::new(child),
                stdin: Mutex::new(stdin),
                stdout: Mutex::new(BufReader::new(stdout).lines()),
            }),
            next_id: Mutex::new(1),
            pending: Mutex::new(HashMap::new()),
        });
        client.initialize().await?;
        Ok(client)
    }

    pub async fn discover_tools(self: &Arc<Self>) -> Result<Vec<McpTool>, McpError> {
        let result = self.request("tools/list", json!({})).await?;
        let listed: ToolList = serde_json::from_value(result)
            .map_err(|error| McpError::Protocol(format!("invalid tools/list result: {error}")))?;
        Ok(listed
            .tools
            .into_iter()
            .map(|definition| McpTool {
                client: Arc::clone(self),
                remote_name: definition.name.clone(),
                name: definition.name,
                description: definition.description.unwrap_or_default(),
                parameters: definition.input_schema,
            })
            .collect())
    }

    async fn initialize(&self) -> Result<(), McpError> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": { "name": "agent-r", "version": env!("CARGO_PKG_VERSION") }
            }),
        )
        .await?;
        self.notify("notifications/initialized", json!({})).await
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        let payload = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.write(payload).await
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = {
            let mut next = self.next_id.lock().await;
            let id = *next;
            *next += 1;
            id
        };
        self.write(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await?;
        loop {
            if let Some(response) = self.pending.lock().await.remove(&id) {
                return resolve(response);
            }
            let mut stdout = self.transport.stdout.lock().await;
            // Another caller may have buffered our reply while we waited for
            // the reader lock.
            if let Some(response) = self.pending.lock().await.remove(&id) {
                return resolve(response);
            }
            let Some(line) = stdout.next_line().await.map_err(McpError::Transport)? else {
                return Err(McpError::Closed);
            };
            let response: JsonRpcResponse = serde_json::from_str(&line)
                .map_err(|error| McpError::Protocol(format!("invalid JSON-RPC line: {error}")))?;
            match response.id {
                Some(response_id) if response_id == id => return resolve(response),
                // A reply to a different in-flight request is buffered for its
                // caller instead of being dropped.
                Some(response_id) => {
                    self.pending.lock().await.insert(response_id, response);
                }
                // Server notifications carry no id and are ignored.
                None => {}
            }
        }
    }

    async fn write(&self, payload: Value) -> Result<(), McpError> {
        let bytes = serde_json::to_vec(&payload).map_err(|error| {
            McpError::Protocol(format!("request serialization failed: {error}"))
        })?;
        let mut stdin = self.transport.stdin.lock().await;
        stdin.write_all(&bytes).await.map_err(McpError::Transport)?;
        stdin.write_all(b"\n").await.map_err(McpError::Transport)?;
        stdin.flush().await.map_err(McpError::Transport)
    }
}

#[derive(Deserialize)]
struct ToolList {
    tools: Vec<McpToolDefinition>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct McpToolDefinition {
    name: String,
    description: Option<String>,
    input_schema: Value,
}
#[derive(Deserialize)]
struct JsonRpcResponse {
    id: Option<u64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

fn resolve(response: JsonRpcResponse) -> Result<Value, McpError> {
    if let Some(error) = response.error {
        return Err(McpError::Remote {
            code: error.code,
            message: error.message,
        });
    }
    response
        .result
        .ok_or_else(|| McpError::Protocol("response had neither result nor error".into()))
}
#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

/// A discovered remote tool. Its name is namespaced when registered with a bridge.
pub struct McpTool {
    client: Arc<McpClient>,
    remote_name: String,
    name: String,
    description: String,
    parameters: Value,
}

impl McpTool {
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }
    fn namespaced(mut self, namespace: &str) -> Self {
        self.name = format!("{namespace}.{}", self.name);
        self
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> Value {
        self.parameters.clone()
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let call = self.client.request(
            "tools/call",
            json!({"name": self.remote_name, "arguments": args}),
        );
        let result = tokio::select! {
            _ = ctx.cancellation.cancelled() => return Err(ToolError::Cancelled),
            result = call => result.map_err(|error| ToolError::Execution(error.to_string()))?,
        };
        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let text = result
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| result.to_string());
        if is_error {
            return Err(ToolError::Execution(text));
        }
        Ok(ToolOutput {
            text,
            payload: result.get("structuredContent").cloned(),
        })
    }
}

/// Registers discovered tools under `namespace.tool_name`, avoiding collisions
/// with built-ins and other MCP servers.
pub struct McpToolBridge;
impl McpToolBridge {
    pub async fn register(
        registry: ToolRegistry,
        namespace: &str,
        client: &Arc<McpClient>,
    ) -> Result<ToolRegistry, McpError> {
        let mut registry = registry;
        for tool in client.discover_tools().await? {
            registry = registry.register(
                tool.namespaced(namespace),
                ToolSource::Mcp {
                    server: namespace.into(),
                },
            )?;
        }
        Ok(registry)
    }
}
