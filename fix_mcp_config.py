import json
import os
import sys

config_path = os.path.expandvars(r"%APPDATA%\Antigravity\mcp_config.json")

print(f"Target file: {config_path}")

# The desired configuration
config = {
    "mcpServers": {
        "claude-mem": {
            "command": "npx",
            "args": ["-y", "tsx", r"C:\mcp-servers\claude-mem\src\servers\mcp-server.ts"],
            "env": {"MEM_PATH": r"C:\mcp-servers\claude-mem\db\mem.db", "DETAIL_LEVEL": "HIGH", "MAX_SUMMARY_TOKENS": "1500"},
        },
        "sequential-thinking": {
            "command": r"C:\Users\nicor\.bun\bin\bun.exe",
            "args": ["run", "dist/index.js"],
            "cwd": r"C:\Users\nicor\.gemini\mcp-servers\node_modules\@modelcontextprotocol\server-sequential-thinking",
        },
    }
}

try:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Successfully wrote valid JSON.")

    # Verify reading it back
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print("Successfully verified JSON integrity.")
    print(json.dumps(loaded, indent=2))
except Exception as e:
    print(f"Error: {e}")
