//! Optional built-in tools. All filesystem paths are confined to `ToolContext::workspace_root`.
use agent_core::{Tool, ToolContext, ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};

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
        let text = response
            .text()
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolOutput {
            text,
            payload: Some(json!({"status": status.as_u16(), "success": status.is_success()})),
        })
    }
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
        let output = tokio::select! { _=ctx.cancellation.cancelled()=>return Err(ToolError::Cancelled), result=tokio::process::Command::new(program).args(argv?).current_dir(&ctx.workspace_root).output()=>result.map_err(|e|ToolError::Execution(e.to_string()))? };
        let text = format!(
            "exit: {}\nstdout:\n{}\nstderr:\n{}",
            output
                .status
                .code()
                .map_or_else(|| "signal".into(), |n| n.to_string()),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(ToolOutput {
            text,
            payload: Some(
                json!({"success":output.status.success(),"exit_code":output.status.code()}),
            ),
        })
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
}
