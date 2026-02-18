# Chinese-SimpleQA Environment

A Chinese-language factuality evaluation benchmark for OpenReward, containing 3,000 short-form Chinese Q&A pairs with LLM-based grading.

## Overview

Chinese-SimpleQA is an OpenReward environment that evaluates large language models' ability to answer short factual questions in Chinese. Based on OpenAI's SimpleQA methodology, it uses LLM-based grading to classify answers as:

- **CORRECT**: Semantically equivalent to reference answer
- **INCORRECT**: Wrong or contradictory information
- **NOT_ATTEMPTED**: Model indicates uncertainty or doesn't answer

## Dataset Details

- **Size**: 3,000 tasks
- **Language**: Chinese (Simplified)
- **Topics**: 6 major categories with 99 fine-grained subtopics
  - 中华文化 (Chinese Culture)
  - Humanities
  - Engineering, Technology, and Applied Sciences
  - Life, Art, and Culture
  - Society
  - Natural Science
- **Source**: [OpenStellarTeam/Chinese-SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA)
- **Paper**: [arXiv:2411.07140](https://arxiv.org/abs/2411.07140)

## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EnvCommons/chinesimpleqa.git
   cd chinesimpleqa
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**:
   ```python
   from datasets import load_dataset
   ds = load_dataset("OpenStellarTeam/Chinese-SimpleQA", split="train")
   ds.to_pandas()[["id", "question", "answer"]].to_parquet("chinese_simpleqa.parquet")
   ```

4. **Start the server**:
   ```bash
   python server.py
   ```

### Docker

```bash
docker build -t chinesimpleqa .
docker run -p 8080:8080 \
  -v $(pwd):/orwd_data/chinesimpleqa \
  chinesimpleqa
```

## Usage

### Testing with OpenAI Models

```bash
export OPENAI_API_KEY=sk-...
python test_agent_openai.py
```

### Testing with Anthropic Models

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...  # Still needed for grading
python test_agent_anthropic.py
```

### Programmatic Usage

```python
from openreward import AsyncOpenReward

or_client = AsyncOpenReward()
environment = or_client.environments.get(name="EnvCommons/chinesimpleqa")

# List tasks
tasks = await environment.list_tasks(split="test")
print(f"Found {len(tasks)} tasks")

# Run a task
async with environment.session(
    task=tasks[0],
    secrets={"openai_api_key": "sk-..."}
) as session:
    prompt = await session.get_prompt()
    print(prompt)  # Chinese question

    # Submit answer
    result = await session.call_tool(
        "submit_answer",
        {"answer": "你的中文答案"}
    )
    print(f"Reward: {result.reward}")
    print(f"Classification: {result.metadata['classification']}")
```

## Environment Structure

- **Split**: `test` (single split with 3,000 tasks)
- **Tools**:
  - `submit_answer(answer: str)`: Submit Chinese answer for evaluation
- **Grading**: LLM-based using `gpt-5-mini`
- **Reward**: Binary (1.0 for correct, 0.0 for incorrect/not_attempted)

## Example Task

```json
{
  "id": "97e7f58a3b154facaa3a5c64d678c7bf",
  "question": "伏兔穴所属的经脉是什么？",
  "answer": "足阳明胃经"
}
```

**Translation**:
- Question: "What meridian does the Futu acupoint belong to?"
- Answer: "Stomach Meridian of Foot-Yangming"

## Grading Methodology

Answers are evaluated by an LLM (gpt-5-mini) that:

1. **Accepts semantic equivalence**: Synonyms and paraphrasing are accepted
2. **Focuses on meaning**: Character-level differences are tolerated
3. **Detects non-attempts**: Phrases like "我不知道" (I don't know) are classified separately
4. **Returns structured output**: Classification + reasoning + reward

### Grading Accuracy

- Estimated grader accuracy: ~94.4% (based on SimpleQA framework)
- Benchmark error rate: ~3% inherent error

## Data Requirements

See [DATA_UPLOAD.md](DATA_UPLOAD.md) for instructions on uploading the dataset to OpenReward cloud storage.

**Required file**: `/orwd_data/chinesimpleqa/chinese_simpleqa.parquet`

## Files

- `chinesimpleqa.py`: Main environment class with grading logic
- `server.py`: Minimal server wrapper (8 lines)
- `test_agent_openai.py`: OpenAI test client
- `test_agent_anthropic.py`: Anthropic test client
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container definition
- `DATA_UPLOAD.md`: Data upload instructions

## Development

### Running Tests

```bash
# Syntax check
python -m py_compile *.py

# Start server
python server.py

# Test with agent (in another terminal)
export OPENAI_API_KEY=sk-...
python test_agent_openai.py
```

### Docker Build

```bash
docker build -t chinesimpleqa:test .
docker run -p 8080:8080 \
  -v $(pwd):/orwd_data/chinesimpleqa \
  chinesimpleqa:test
```

## Performance Characteristics

- **Evaluation Cost**: ~$0.30 per full benchmark run (3,000 tasks × ~$0.0001/grading)
- **Latency**: ~1-2 seconds per answer (LLM grading)
- **Model Requirements**: Any model with Chinese language support

## Attribution

**Dataset**: OpenStellarTeam/Chinese-SimpleQA
**Paper**: Chinese SimpleQA: A Chinese Factuality Evaluation Benchmark (arXiv:2411.07140)
**Methodology**: Based on OpenAI's SimpleQA framework
**Environment**: Built for OpenReward platform

## License

See dataset license on [HuggingFace](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA).

## Citation

```bibtex
@misc{wu2024chinesesimpleqa,
  title={Chinese SimpleQA: A Chinese Factuality Evaluation Benchmark},
  author={Wu, Yixuan and others},
  year={2024},
  eprint={2411.07140},
  archivePrefix={arXiv}
}
```

## Contact

For issues or questions:
- GitHub: https://github.com/EnvCommons/chinesimpleqa
- OpenReward: https://openreward.ai
