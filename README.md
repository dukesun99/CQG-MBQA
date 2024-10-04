# CQG-MBQA

## Prerequisites
- Anaconda (recommended, but optional)
- OpenAI API key set up in environment variables (see [OpenAI Quickstart](https://platform.openai.com/docs/quickstart))
- Sufficient OpenAI API credits
- GPU with CUDA support for faster encoding and training (recommended, but optional)

## Setup

### Create and Activate Conda Environment
```bash
conda create -n cqgmbqa python=3.9
conda activate cqgmbqa
pip install -r requirements.txt
```
Ensure `OPENAI_API_KEY` is set in your environment variables before proceeding.

## Running the Pipeline with Default Configurations

### MEDI2 Dataset
The MEDI2 dataset is available on [Hugging Face](https://huggingface.co/datasets/GritLM/MEDI2/tree/main).

To use the dataset:
1. Clone the repository
2. Pull LFS files
3. Run the preprocessing steps

Or just access the preprocessed JSON data at `data/medi2_documents.json`

### Execute the Pipeline
```bash
python framework/MEDI2.py
```

This creates an `output` folder in the working directory, containing:
- Generated questions from the pipeline
- Model checkpoint trained on the MEDI2 dataset

### Use pre-trained model for inference
Alternatively, we provide the pre-trained model used in our experiments in the `checkpoints` folder. You can use the model to perform inference with the MTEB benchmark.

### MTEB Benchmark Evaluation
To evaluate using the MTEB benchmark:
1. Import `MBQAMTEBModelWrapper` from `framework.mbqa_model`
2. Install our modified benchmark package:
   ```bash
   cd mteb
   pip install .
   ```
Our modification includes a cognitive load score for STS tasks.

## Customizing the Pipeline for Your Dataset

Our framework is adaptable to various datasets with appropriate configurations. You can fine-tune the model on your text corpus for optimal embedding performance.

### Dataset Preprocessing
Prepare your dataset as a JSON file containing a list of strings, where each string represents a document:
```json
[
    "This is a test document",
    "This is another test document"
]
```


### Pipeline Configuration
- For dataset configuration: Refer to `framework/MEDI2.py`
- For CQG generation parameters: See `framework/cqg.py`
- For model and training parameters: Check `framework/mbqa.py`

Key parameters affecting the number of generated questions:
- `k`: Number of clusters in k-means clustering
- `t`: Top `t` questions selected from each cluster
- `θ`: Threshold for deduplication process

Tip: Start by setting `k` based on your document count. A good MBQA model typically requires an average of at least 100 documents per cluster.

### Using Your Trained Model
1. Locate your trained model and generated questions in the `output` folder
2. Import `MBQAMTEBModelWrapper` from `framework.mbqa_model` for inference

The model functions similarly to a sentence-transformer model, utilizing the `encode` method for embeddings.

