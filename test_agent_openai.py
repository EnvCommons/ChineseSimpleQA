import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


async def main() -> None:
    """Test the Chinese-SimpleQA environment with OpenAI models."""
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to local server
    environment = or_client.environments.get(
        name="local/ChineseSimpleQA", base_url="http://localhost:8080"
    )

    # Get tasks and tools
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

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
        input_list = [{"role": "user", "content": prompt}]

        turn = 0
        while not finished:
            turn += 1
            print(f"\n[Turn {turn}] Calling model...")

            # Use modern Responses API
            response = await oai_client.responses.create(
                model=MODEL_NAME, tools=tools, input=input_list
            )

            # Add model output to conversation
            input_list += response.output

            # Process function calls
            for item in response.output:
                if item.type == "function_call":
                    print(f"  Tool called: {item.name}")
                    print(f"  Arguments: {item.arguments}")

                    # Call environment tool
                    tool_result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward

                    # Add tool result to conversation
                    tool_output = (
                        tool_result.blocks[0].text if tool_result.blocks else ""
                    )
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_output,
                        }
                    )

                    print(f"  Reward: {reward:.3f}")
                    print(f"  Finished: {finished}")

                    if tool_result.blocks:
                        print(f"\n{tool_result.blocks[0].text}")

                    if finished:
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
