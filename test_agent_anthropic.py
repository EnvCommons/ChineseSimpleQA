import asyncio
import os

from anthropic import AsyncAnthropic
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "claude-sonnet-4-5-20250929")
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


async def main() -> None:
    """Test the Chinese-SimpleQA environment with Anthropic models."""
    or_client = AsyncOpenReward()
    ant_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Connect to local server
    environment = or_client.environments.get(
        name="local/ChineseSimpleQA", base_url="http://localhost:8080"
    )

    # Get tasks and tools
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="anthropic")

    print(f"Loaded {len(tasks)} tasks from Chinese-SimpleQA")
    print(f"Testing with model: {MODEL_NAME}")
    print("-" * 60)

    # Test first task
    task = tasks[0]
    print(f"\nTask ID: {task.task_spec['id']}")
    print(f"Question: {task.task_spec['question']}")
    print(f"Reference Answer: {task.task_spec['answer']}")
    print("-" * 60)

    finished = False

    async with environment.session(
        task=task, secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        prompt = await session.get_prompt()

        # Initialize conversation with user prompt
        messages = [{"role": "user", "content": prompt}]

        turn = 0
        while not finished:
            turn += 1
            print(f"\n[Turn {turn}] Calling model...")

            # Use Anthropic Messages API
            message = await ant_client.messages.create(
                model=MODEL_NAME, max_tokens=4096, tools=tools, messages=messages
            )

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": message.content})

            print(f"  Stop reason: {message.stop_reason}")

            # Process tool use if present
            if message.stop_reason == "tool_use":
                # Find tool use block
                tool_use = next(
                    block for block in message.content if block.type == "tool_use"
                )

                print(f"  Tool called: {tool_use.name}")
                print(f"  Arguments: {tool_use.input}")

                # Call environment tool
                tool_result = await session.call_tool(tool_use.name, tool_use.input)

                finished = tool_result.finished
                reward = tool_result.reward

                # Add tool result to conversation
                tool_output = tool_result.blocks[0].text if tool_result.blocks else ""
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_output,
                            }
                        ],
                    }
                )

                print(f"  Reward: {reward:.3f}")
                print(f"  Finished: {finished}")

                if tool_result.blocks:
                    print(f"\n{tool_result.blocks[0].text}")

                if finished:
                    break

            elif message.stop_reason == "end_turn":
                # Model finished without calling tool (shouldn't happen in this env)
                print("\n⚠️  Model finished without calling tool")
                break

            # Safety check: prevent infinite loops
            if turn > 10:
                print("\n⚠️  Maximum turns reached")
                break

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
