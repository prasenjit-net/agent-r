//! Optional built-in tools. All filesystem paths are confined to `ToolContext::workspace_root`.
use agent_core::{Tool, ToolContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::process::Stdio;

/// Response bodies are capped so a large or unbounded download cannot blow up
/// memory or the agent's context budget in one tool result.
#[cfg(feature = "http")]
const MAX_HTTP_BODY_BYTES: usize = 256 * 1024;

#[cfg(feature = "http")]
pub struct HttpGet;
#[cfg(feature = "http")]
#[async_trait]
impl Tool for HttpGet {
    fn name(&self) -> &str {
        "http.get"
    }
    fn description(&self) -> &str {
        "Fetch an HTTP(S) URL and return its response body."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","required":["url"],"properties":{"url":{"type":"string","format":"uri"}}})
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let url = args
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("url is required".into()))?;
        let request = reqwest::get(url);
        let response = tokio::select! { _ = ctx.cancellation.cancelled() => return Err(ToolError::Cancelled), result = request => result.map_err(|e| ToolError::Execution(e.to_string()))? };
        let status = response.status();
        let (mut text, truncated) = capped_body(response, MAX_HTTP_BODY_BYTES, ctx).await?;
        if truncated {
            text.push_str(&format!("\n[truncated after {MAX_HTTP_BODY_BYTES} bytes]"));
        }
        Ok(ToolOutput {
            text,
            payload: Some(
                json!({"status": status.as_u16(), "success": status.is_success(), "truncated": truncated}),
            ),
        })
    }
}

/// Streams the response body, discarding bytes past `cap` instead of
/// buffering the whole thing before truncating.
#[cfg(feature = "http")]
async fn capped_body(
    response: reqwest::Response,
    cap: usize,
    ctx: &ToolContext,
) -> Result<(String, bool), ToolError> {
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::with_capacity(cap.min(8192));
    let mut truncated = false;
    loop {
        let next = tokio::select! {
            _ = ctx.cancellation.cancelled() => return Err(ToolError::Cancelled),
            next = stream.next() => next,
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|e| ToolError::Execution(e.to_string()))?;
        if buffer.len() >= cap {
            truncated = true;
            continue;
        }
        let remaining = cap - buffer.len();
        if chunk.len() > remaining {
            buffer.extend_from_slice(&chunk[..remaining]);
            truncated = true;
        } else {
            buffer.extend_from_slice(&chunk);
        }
    }
    Ok((String::from_utf8_lossy(&buffer).into_owned(), truncated))
}

use std::path::{Component, Path, PathBuf};

async fn safe_path(root: &Path, requested: &str) -> Result<PathBuf, ToolError> {
    let path = Path::new(requested);
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(ToolError::InvalidArguments(
            "path must be relative and must not escape the workspace".into(),
        ));
    }
    let root = tokio::fs::canonicalize(root)
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    // Canonicalization resolves symlinks, so a link inside the workspace that
    // points outside it cannot escape confinement.
    let resolved = tokio::fs::canonicalize(root.join(path))
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    if !resolved.starts_with(&root) {
        return Err(ToolError::InvalidArguments(
            "path must be relative and must not escape the workspace".into(),
        ));
    }
    Ok(resolved)
}

#[cfg(feature = "fs")]
pub struct ReadFile;
#[cfg(feature = "fs")]
#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "filesystem.read_file"
    }
    fn description(&self) -> &str {
        "Read a UTF-8 file relative to the workspace root."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","required":["path"],"properties":{"path":{"type":"string"}}})
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("path is required".into()))?;
        let contents = tokio::fs::read_to_string(safe_path(&ctx.workspace_root, path).await?)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolOutput {
            text: contents,
            payload: None,
        })
    }
}

#[cfg(feature = "fs")]
pub struct ListDir;
#[cfg(feature = "fs")]
#[async_trait]
impl Tool for ListDir {
    fn name(&self) -> &str {
        "filesystem.list_dir"
    }
    fn description(&self) -> &str {
        "List a directory relative to the workspace root."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string"}}})
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
        let mut entries = tokio::fs::read_dir(safe_path(&ctx.workspace_root, path).await?)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let mut names = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?
        {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        Ok(ToolOutput {
            text: names.join("\n"),
            payload: Some(json!(names)),
        })
    }
}

/// Captured stdout/stderr are capped so a runaway or chatty command cannot
/// blow up memory or the agent's context budget in one tool result.
#[cfg(feature = "shell")]
const MAX_COMMAND_OUTPUT_BYTES: usize = 64 * 1024;

#[cfg(feature = "shell")]
pub struct RunCommand;
#[cfg(feature = "shell")]
#[async_trait]
impl Tool for RunCommand {
    fn name(&self) -> &str {
        "shell.run_command"
    }
    fn description(&self) -> &str {
        "Run an executable and arguments in the workspace; no shell interpolation."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","required":["program"],"properties":{"program":{"type":"string"},"args":{"type":"array","items":{"type":"string"}}}})
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let program = args
            .get("program")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments("program is required".into()))?;
        let values = args
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let argv: Result<Vec<_>, _> = values
            .iter()
            .map(|v| {
                v.as_str().ok_or_else(|| {
                    ToolError::InvalidArguments("args must contain only strings".into())
                })
            })
            .collect();
        let mut child = tokio::process::Command::new(program)
            .args(argv?)
            .current_dir(&ctx.workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Otherwise a cancelled call leaves the process running in the background.
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        // Both streams must be drained concurrently (not sequentially, and not
        // after `wait()`): if a command writes more than the OS pipe buffer
        // holds, it blocks on write() until something reads, so waiting for
        // exit before draining would deadlock.
        let (stdout_result, stderr_result) = tokio::select! {
            _ = ctx.cancellation.cancelled() => return Err(ToolError::Cancelled),
            result = async {
                tokio::join!(
                    capped_read(stdout, MAX_COMMAND_OUTPUT_BYTES),
                    capped_read(stderr, MAX_COMMAND_OUTPUT_BYTES),
                )
            } => result,
        };
        let (stdout_bytes, stdout_truncated) =
            stdout_result.map_err(|e| ToolError::Execution(e.to_string()))?;
        let (stderr_bytes, stderr_truncated) =
            stderr_result.map_err(|e| ToolError::Execution(e.to_string()))?;
        let status = tokio::select! {
            _ = ctx.cancellation.cancelled() => return Err(ToolError::Cancelled),
            result = child.wait() => result.map_err(|e| ToolError::Execution(e.to_string()))?,
        };
        let text = format!(
            "exit: {}\nstdout:\n{}{}\nstderr:\n{}{}",
            status
                .code()
                .map_or_else(|| "signal".into(), |n| n.to_string()),
            String::from_utf8_lossy(&stdout_bytes),
            if stdout_truncated {
                "\n[stdout truncated]"
            } else {
                ""
            },
            String::from_utf8_lossy(&stderr_bytes),
            if stderr_truncated {
                "\n[stderr truncated]"
            } else {
                ""
            },
        );
        Ok(ToolOutput {
            text,
            payload: Some(json!({
                "success": status.success(),
                "exit_code": status.code(),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            })),
        })
    }
}

#[cfg(feature = "shell")]
async fn capped_read(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    cap: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    use tokio::io::AsyncReadExt;
    let mut buffer = Vec::with_capacity(cap.min(8192));
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        if buffer.len() >= cap {
            truncated = true;
            continue;
        }
        let remaining = cap - buffer.len();
        if n > remaining {
            buffer.extend_from_slice(&chunk[..remaining]);
            truncated = true;
        } else {
            buffer.extend_from_slice(&chunk[..n]);
        }
    }
    Ok((buffer, truncated))
}

/// A preset of built-in tools and the matching allow policy for a common
/// agent role. Each bundle is registered from tools that already exist under
/// this crate's default feature set; it is a convenience over manual
/// registration, not a different execution path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityBundle {
    /// Read, list, and run commands in a workspace — for a coding agent.
    Coding,
    /// Fetch URLs and run commands — for an operational health-check agent.
    Ops,
}

impl CapabilityBundle {
    pub fn tool_names(&self) -> &'static [&'static str] {
        match self {
            CapabilityBundle::Coding => &[
                "filesystem.read_file",
                "filesystem.list_dir",
                "shell.run_command",
            ],
            CapabilityBundle::Ops => &["http.get", "shell.run_command"],
        }
    }

    pub fn register(
        &self,
        registry: agent_core::ToolRegistry,
    ) -> Result<agent_core::ToolRegistry, agent_core::AgentError> {
        match self {
            #[cfg(all(feature = "fs", feature = "shell"))]
            CapabilityBundle::Coding => registry
                .register(ReadFile, agent_core::ToolSource::Builtin)?
                .register(ListDir, agent_core::ToolSource::Builtin)?
                .register(RunCommand, agent_core::ToolSource::Builtin),
            #[cfg(all(feature = "http", feature = "shell"))]
            CapabilityBundle::Ops => registry
                .register(HttpGet, agent_core::ToolSource::Builtin)?
                .register(RunCommand, agent_core::ToolSource::Builtin),
            #[allow(unreachable_patterns)]
            _ => Ok(registry),
        }
    }

    /// An allowlist policy granting exactly this bundle's tools. Combine with
    /// `ToolPolicy::default()`'s `RequireApproval` default for anything else.
    pub fn allow_policy(&self) -> agent_core::ToolPolicy {
        let mut policy = agent_core::ToolPolicy::default();
        for name in self.tool_names() {
            policy
                .per_tool
                .insert((*name).to_string(), agent_core::ToolPermission::Allow);
        }
        policy
    }
}

#[cfg(all(test, feature = "fs", unix))]
mod tests {
    use super::*;
    use agent_core::Tool;
    use tokio_util::sync::CancellationToken;

    fn context_for(workspace: &Path) -> ToolContext {
        ToolContext {
            session_id: uuid::Uuid::new_v4(),
            workspace_root: workspace.to_path_buf(),
            cancellation: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn read_file_rejects_parent_dir_and_absolute_paths() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = context_for(workspace.path());
        for path in ["../secret.txt", "/etc/hosts"] {
            let result = ReadFile.run(json!({ "path": path }), &ctx).await;
            assert!(
                matches!(result, Err(ToolError::InvalidArguments(_))),
                "{path} must be rejected"
            );
        }
    }

    #[tokio::test]
    async fn read_file_rejects_symlinks_that_leave_the_workspace() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("ok.txt"), "ok").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            workspace.path().join("link.txt"),
        )
        .unwrap();
        let ctx = context_for(workspace.path());

        let escape = ReadFile.run(json!({"path": "link.txt"}), &ctx).await;
        assert!(matches!(escape, Err(ToolError::InvalidArguments(_))));
        let allowed = ReadFile.run(json!({"path": "ok.txt"}), &ctx).await.unwrap();
        assert_eq!(allowed.text, "ok");
    }

    #[cfg(feature = "shell")]
    #[tokio::test]
    async fn run_command_caps_captured_output() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = context_for(workspace.path());
        // `yes` streams output indefinitely; without draining concurrently
        // with capping, this would either hang or exhaust memory.
        let output = RunCommand
            .run(
                json!({"program": "sh", "args": ["-c", "yes | head -c 200000"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(output.payload.unwrap()["stdout_truncated"] == true);
        assert!(output.text.len() < 200_000);
    }

    #[cfg(feature = "shell")]
    #[tokio::test]
    async fn run_command_reports_exit_status() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = context_for(workspace.path());
        let output = RunCommand
            .run(json!({"program": "sh", "args": ["-c", "exit 7"]}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.payload.unwrap()["exit_code"], 7);
    }

    #[cfg(all(feature = "fs", feature = "shell"))]
    #[test]
    fn coding_bundle_registers_its_advertised_tools() {
        let registry = CapabilityBundle::Coding
            .register(agent_core::ToolRegistry::new())
            .unwrap();
        for name in CapabilityBundle::Coding.tool_names() {
            assert!(registry.get(name).is_some(), "missing tool: {name}");
        }
        let policy = CapabilityBundle::Coding.allow_policy();
        for name in CapabilityBundle::Coding.tool_names() {
            assert!(matches!(
                policy.permission(name),
                agent_core::ToolPermission::Allow
            ));
        }
    }
}
