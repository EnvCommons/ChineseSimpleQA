# Data Upload Requirements for Chinese-SimpleQA

## Overview

This environment requires the Chinese-SimpleQA dataset to be uploaded to OpenReward cloud storage at `/orwd_data/chinesimpleqa/`. The dataset is NOT included in the Docker image to keep it lightweight and deployable.

## Directory Structure

```
/orwd_data/
└── chinesimpleqa/
    └── chinese_simpleqa.parquet
```

## File Description

- **chinese_simpleqa.parquet**: Main dataset containing 3,000 Chinese Q&A task pairs
  - Size: ~500 KB
  - Columns: `id`, `question`, `answer`
  - Format: Apache Parquet (compressed)

## How to Generate the Data File

Use the following Python script to download and prepare the dataset:

```python
from datasets import load_dataset
import pandas as pd

# Load dataset from HuggingFace
print("Loading Chinese-SimpleQA dataset...")
ds = load_dataset("OpenStellarTeam/Chinese-SimpleQA", split="train")

# Convert to pandas and keep only necessary columns
df = ds.to_pandas()[["id", "question", "answer"]]

# Save as parquet
df.to_parquet("chinese_simpleqa.parquet")

print(f"Saved {len(df)} tasks to chinese_simpleqa.parquet")
```

## Upload Instructions

1. **Generate the parquet file locally** using the script above
2. **Upload to OpenReward** at https://openreward.ai
3. **Configure namespace**: `EnvCommons/chinesimpleqa`
4. **Upload path**: `/orwd_data/chinesimpleqa/chinese_simpleqa.parquet`

## Verification

After upload, verify the data is accessible:
- The file should be readable at `/orwd_data/chinesimpleqa/chinese_simpleqa.parquet`
- Should contain exactly 3,000 records
- Each record has fields: `id`, `question`, `answer`

## Dataset Attribution

**Dataset**: [OpenStellarTeam/Chinese-SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA)
**Paper**: [Chinese SimpleQA: A Chinese Factuality Evaluation Benchmark](https://arxiv.org/abs/2411.07140)
**License**: Check HuggingFace dataset page for license details

## Notes

- The dataset is based on OpenAI's SimpleQA methodology
- Contains 3,000 short-form Chinese factual questions
- Covers 6 major topics with 99 fine-grained subtopics
- Questions are designed for LLM factuality evaluation
