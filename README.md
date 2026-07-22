# ChineseSimpleQA

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ChineseSimpleQA) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA)

## Description

ChineseSimpleQA is an environment for evaluating the factual accuracy of language models on short-form Chinese questions. Based on OpenAI's SimpleQA methodology adapted for Chinese, it contains 3,000 factual questions spanning 6 major topics and 99 fine-grained subtopics. Answers are graded by an LLM (gpt-5-mini) that classifies responses as CORRECT, INCORRECT, or NOT_ATTEMPTED based on semantic equivalence to a reference answer.

## Capabilities

- Answering short-form factual questions in Chinese
- Demonstrating knowledge across a broad range of Chinese-language topics including Chinese culture, humanities, natural science, engineering and technology, society, and life, art, and culture
- Providing precise, semantically accurate responses

## Compute Requirements

This is a single-turn environment with no sandbox compute requirements. The agent receives a question and submits an answer.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are 3,000 tasks in the `test` split. Each task presents a short factual question in Chinese and expects a Chinese-language answer. The questions cover 6 major topics with 99 fine-grained subtopics:

- Chinese Culture
- Humanities
- Engineering, Technology, and Applied Sciences
- Life, Art, and Culture
- Society
- Natural Science

## Reward Structure

This is a sparse, binary reward environment. The reward is determined by LLM-based grading (gpt-5-mini) that classifies the submitted answer into one of three categories:

- **CORRECT** (reward = 1.0): The answer is semantically equivalent to the reference answer. Synonyms, paraphrasing, and different wordings that convey the same meaning are accepted.
- **INCORRECT** (reward = 0.0): The answer contradicts the reference answer or provides wrong information.
- **NOT_ATTEMPTED** (reward = 0.0): The answer indicates the agent does not know or is empty/evasive.

The grader focuses on semantic meaning in Chinese rather than exact character matching.

## Data

The dataset consists of 3,000 Chinese factual question-answer pairs stored in `chinese_simpleqa.parquet`. The data is sourced from the [OpenStellarTeam/Chinese-SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA) HuggingFace dataset. Each record contains an `id`, `question`, and `answer` field.

## Tools

ChineseSimpleQA exposes no tools to the agent. The rollout ends as soon as the model emits a plain assistant message; that message is graded in Chinese by gpt-5-mini against the reference answer.

## Time Horizon

This is a single-turn environment. The agent receives one question and replies with its answer as an ordinary message.

## Environment Difficulty

The original paper evaluates models on Chinese SimpleQA (Correct %):

| Model | Correct |
|-------|---------|
| o1-preview | 63.8% |
| Doubao-pro-32k | 61.9% |
| GPT-4o | 59.3% |
| GLM-4-Plus | 58.7% |
| Qwen2.5-72B | 48.4% |
| Claude-3.5-Sonnet | 46.2% |
| GPT-4 | 45.4% |
| LLaMA3.1-70B | 38.3% |
| GPT-4o-mini | 37.6% |

## Other Environment Requirements

This environment requires an OpenAI API key for LLM-based answer grading.

## Safety

ChineseSimpleQA is a factual question-answering benchmark. The agent only interacts with the environment by submitting text answers to factual questions. There are no safety concerns associated with this environment, as the agent has no access to external systems, file systems, or the internet during evaluation.

## Citations

```bibtex
@article{he2024chinesesimpleqa,
  author    = {Yancheng He and Shilong Li and Jiaheng Liu and Yingshui Tan and Weixun Wang and Hui Huang and Xingyuan Bu and Hangyu Guo and Chengwei Hu and Boren Zheng and Zhuoran Lin and Xuepeng Liu and Dekai Sun and Shirong Lin and Zhicheng Zheng and Xiaoyong Zhu and Wenbo Su and Bo Zheng},
  title     = {Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models},
  journal   = {arXiv preprint arXiv:2411.07140},
  year      = {2024},
  url       = {https://arxiv.org/abs/2411.07140}
}
```
