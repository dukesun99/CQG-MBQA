# Contrastive Question Generation

## Overview
The Contrastive Question Generation (CQG) module is designed to generate high-quality discriminative questions from a given corpus of text. It utilizes a contrastive learning-like approach and probing based post-processing to ensure that the generated questions are both diverse and relevant to the input text.

## Usage
The CQG module could be called as follows:

```python
from cqg import ContrastiveQuestionGeneration

# Load the documents first
with open("data/dataset.json", "r") as f:
    doc_texts = json.load(f)

cqg = ContrastiveQuestionGeneration(
    corpus=doc_texts,
    temp_folder="./temp",
    name="DatasetName",
    encoder="WhereIsAI/UAE-Large-V1",
    LLM="gpt-4o-mini",
    k=5000,
    n_pos=6,
    n_hard_neg=18,
    n_easy_neg=18,
    p_pos=5,
    p_hard_neg=3,
    p_easy_neg=2,
    theta=0.8,
    t=4,
    seed=42,
    openai_api_key=None,
    device="cuda"
)

cqg.generate_questions()
```
Arguments for the CQG class:
- `corpus`: (Required) List of text documents to generate questions from.

- `temp_folder`: (Default: "./temp") Directory for storing temporary files during processing. You can delete this folder after the MBQA training process is complete.

- `name`: Name for the current run, used for resuming the process. If not provided, a random name is generated.

- `k`: (Default: 5000) **Number of clusters for K-means clustering $k$**. This should be adjusted based on the size of the corpus and the desired number of dimensions of the interpretable embedding space.

- `theta`: (Default: 0.8) **Threshold for cosine similarity when deduplicating questions $t$**. This affects the number of questions (dimensions) generated.

- `t`: (Default: 4) **Number of top questions to select per cluster $t$**. This affects the number of questions (dimensions) generated.

- `encoder`: (Default: "WhereIsAI/UAE-Large-V1") Model used for encoding text $Enc$.

- `LLM`: (Default: "gpt-4o-mini") Language model used for generating questions $LLM$.

- `n_pos`: (Default: 6) Number of positive samples per cluster $n_{p}$.

- `n_hard_neg`: (Default: 18) Number of hard negative samples per cluster $n_{h}$.

- `n_easy_neg`: (Default: 18) Number of easy negative samples per cluster $n_{e}$.

- `p_pos`: (Default: 5) Number of positive probes per cluster $p_{p}$.

- `p_hard_neg`: (Default: 3) Number of hard negative probes per cluster $p_{h}$.

- `p_easy_neg`: (Default: 2) Number of easy negative probes per cluster $p_{e}$.

- `seed`: (Default: 42) Random seed for reproducibility.

- `openai_api_key`: (Default: None) OpenAI API key. If not provided, it will attempt to use the API key saved in the environment variable `OPENAI_API_KEY`.

- `device`: (Default: "cuda") Device to use for encoding ("cuda" for GPU, "cpu" for CPU).

To adjust the number of questions generated, you can modify the following parameters:
- `k`: Number of clusters for K-means clustering.
- `theta`: Threshold for cosine similarity when deduplicating questions.
- `t`: Number of top questions to select per cluster.

> 💡 These parameters collectively control the number of questions generated. For example, increasing `k` will lead to more clusters, and increasing `t` will select more questions per cluster. Besides, setting a higher `theta` will result in a less strict deduplication of questions, thus also increasing the number of questions.

The `generate_questions()` function will generate the questions and save them to the temp folder. 

> ⚠️ This process may take a while, depending on the size of the corpus and the number of questions generated. Please be patient.