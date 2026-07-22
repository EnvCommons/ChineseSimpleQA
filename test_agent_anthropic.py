"""Agent loop for ChineseSimpleQA (Anthropic Messages API).

ChineseSimpleQA uses a @terminal tool: the grading tool is hidden from the
model, which answers the Chinese question and writes its response as an
ordinary message. The harness sees a message with no tool calls and routes
its text to session.call_terminal_tool(), which grades it against the
reference answer with gpt-5-mini.

Runs against the deployed environment by default; set LOCAL=1 to point at a
local `python server.py` on port 8080.
"""

import asyncio
import os

from anthropic import AsyncAnthropic
from openreward import AsyncOpenReward


def _text_of(message) -> str:
    parts = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


async def main():
    or_client = AsyncOpenReward()
    ant_client = AsyncAnthropic()

    MODEL_NAME = os.environ.get("MODEL_NAME", "claude-sonnet-4-5-20250929")
    ENV_NAME = "GeneralReasoning/ChineseSimpleQA"
    SPLIT = os.environ.get("SPLIT", "test")
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "1"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "40"))
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="anthropic")
    terminal_tool = await environment.terminal_tool()

    print(f"Found {len(tasks)} tasks")
    print(f"Tools visible to the model: {[t['name'] for t in tools]}")
    print(f"Terminal tool (hidden): {terminal_tool}")

    for task in tasks[:NUM_TASKS]:
        print(f"\n=== Task {task.task_spec['id']} ===")

        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY},
        ) as session:
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            print(f"is_assistant_message_final() -> {assistant_ends_rollout}")
            print(f"session.list_tools() -> {[t.name for t in session_tools]}")
            assert "submit_answer" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            messages = [{"role": "user", "content": prompt[0].text}]

            reward = None
            turn = 0

            while turn < MAX_TURNS:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                kwargs = {"model": MODEL_NAME, "max_tokens": 4096, "messages": messages}
                if tools:
                    kwargs["tools"] = tools
                message = await ant_client.messages.create(**kwargs)
                messages.append({"role": "assistant", "content": message.content})

                tool_uses = [b for b in message.content if b.type == "tool_use"]
                if tool_uses:
                    tool_results = []
                    for block in tool_uses:
                        result = await session.call_tool(block.name, block.input)
                        text = result.blocks[0].text if result.blocks else ""
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": text,
                        })
                        print(f"Tool: {block.name}({str(block.input)[:120]})")
                    messages.append({"role": "user", "content": tool_results})
                    continue

                final_message = _text_of(message)
                print(f"Final message: {final_message[:300]}")

                if not assistant_ends_rollout:
                    print("Environment is not terminal-tool style; stopping.")
                    break

                out = await session.call_terminal_tool(final_message)
                reward = out.reward
                print(f"\ncall_terminal_tool -> reward={reward} finished={out.finished}")
                print(out.blocks[0].text[:800])
                break

            print(f"\n=== Task {task.task_spec['id']} reward={reward} turns={turn} ===")


if __name__ == "__main__":
    asyncio.run(main())
