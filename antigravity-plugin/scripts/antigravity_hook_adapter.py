#!/usr/bin/env python3
import json
import hashlib
import os
import subprocess
import sys
import tempfile


def payload():
    try:
        value = json.load(sys.stdin)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def slm(action, data):
    workspace_paths = data.get("workspacePaths")
    env = dict(os.environ)
    if isinstance(workspace_paths, list) and workspace_paths and isinstance(workspace_paths[0], str):
        env["CLAUDE_PROJECT_DIR"] = workspace_paths[0]
    try:
        result = subprocess.run(
            ["slm", "hook", action], input=json.dumps(data), text=True,
            capture_output=True, timeout=8, env=env,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def lifecycle_marker(data):
    conversation_id = data.get("conversationId")
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), "slm-antigravity-" + digest)


def main():
    event = sys.argv[1] if len(sys.argv) == 2 else ""
    data = payload()
    if event == "pre-invocation":
        marker = lifecycle_marker(data)
        if marker and os.path.exists(marker):
            print(json.dumps({"injectSteps": []}))
            return
        context = "\n\n".join(part for part in (slm("mandate", data), slm("start", data)) if part)
        if marker:
            try:
                with open(marker, "x", encoding="utf-8"):
                    pass
            except FileExistsError:
                print(json.dumps({"injectSteps": []}))
                return
        print(json.dumps({"injectSteps": [{"ephemeralMessage": context}]} if context else {"injectSteps": []}))
    elif event == "pre-tool":
        tool_call = data.get("toolCall") if isinstance(data.get("toolCall"), dict) else {}
        slm("before_web", {"tool_input": tool_call.get("args", {})})
        print(json.dumps({"decision": "allow"}))
    elif event == "post-tool":
        tool_call = data.get("toolCall") if isinstance(data.get("toolCall"), dict) else {}
        slm("post_tool_outcome", {
            "tool_name": tool_call.get("name", ""),
            "tool_input": tool_call.get("args", {}),
            "tool_response": data.get("toolResponse", data.get("error", "")),
        })
        print("{}")
    elif event == "stop":
        slm("stop", data)
        slm("stop_outcome", data)
        marker = lifecycle_marker(data)
        if marker:
            try:
                os.remove(marker)
            except OSError:
                pass
        print(json.dumps({"decision": "allow"}))
    else:
        print("{}")


if __name__ == "__main__":
    main()
