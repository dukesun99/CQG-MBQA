# CQG-MBQA: A General Framework for Producing Interpretable Semantic Text Embeddings
Welcome to the repository for our paper "A General Framework for Producing Interpretable Semantic Text Embeddings" (Under Review). 

Key Features:
- 😎 **Interpretable Semantic Text Embeddings**: High quality, highly interpretable semantic text embeddings.
- 🛠️ **General Framework**: Fit for both general domain or task-specific customized interpretable text embeddings.
- 🚀 **Easy to Use**: Clone the repo, install the dependencies, and run the scripts. Easy-peasy!
- 📚 **No Labels Required**: Only requires plain text documents for training with your own dataset.

Our framework allows you to create general domain or task-specific customized interpretable semantic text embeddings.
- For **general domain embeddings** i.e. your downstream task may involve a diverse range of topics which you do not necessarily know the distribution beforehand, you can use our pre-trained model trained with the [MEDI2 dataset](https://huggingface.co/datasets/GritLM/MEDI2), a large collection of diverse texts. To use our pretrained model, please refer to the [Using Pretrained Model Checkpoints](#using-pretrained-model-checkpoints) section.
- For **task-specific embeddings** i.e. your downstream task involves a specific domain or a narrow distribution of topics which you have a corpus (unlabeled) similar to the distribution in the downstream task, you can train your own embeddings using your own dataset. The instructions are given in the [Creating Customized Embeddings Using Your Dataset](#creating-customized-embeddings-using-your-dataset) section.

## Quick Links
- [Installation](#installation)
- [Creating Customized Embeddings Using Your Dataset](#creating-customized-embeddings-using-your-dataset)
- [Using Pretrained Model Checkpoints](#using-pretrained-model-checkpoints)
- [Reproducing Results in Our Paper](#reproducing-results-in-our-paper)
- [Evaluation on MTEB Benchmark](#evaluation-on-mteb-benchmark)
- [Bugs and Issues](#bugs-and-issues)
- [Citation](#citation)

## Installation
### Prerequisites
- **Anaconda**: Recommended, but optional. You may use any other environment management system. We use conda in the following instructions. See [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/).
- [**Git CLI**](https://git-scm.com/downloads) and [**Git LFS**](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) installed. 
- **OpenAI API key**: Required for calling OpenAI APIs. See [OpenAI Quickstart](https://platform.openai.com/docs/quickstart) to get one with sufficient credits. 
- **GPU**: A GPU with CUDA support with at least 6GB VRAM for faster encoding and training is highly recommended, but optional. 

### Setup
Clone our repository and install the dependencies. Repository URL hidden for anonymity. Note this also downloads the LFS files (model checkpoints) automatically so please be patient.
```bash
git clone <repository_url>
cd CQG-MBQA
```


#### Anaconda Environment
First, we need to create and activate a conda environment.
```bash
conda create -n cqgmbqa python=3.9
conda activate cqgmbqa
pip install -r requirements.txt
```

#### OpenAI API Key
Next, ensure `OPENAI_API_KEY` is set in your environment variables before proceeding. 

##### **For Windows Users:**
```cmd
setx OPENAI_API_KEY <your_openai_api_key>
```
To check if the key is set, run:
```cmd
echo %OPENAI_API_KEY%
```

##### **For Linux and MacOS Users:**
Using a text editor of your choice (`nano` in this example), open the `.bashrc` file in your home directory:
```bash
nano ~/.bashrc
```
Add the following line:
```bash
export OPENAI_API_KEY=<your_openai_api_key>
```
Save the file and exit the editor. Apply the changes with:
```bash
source ~/.bashrc
```
To check if the key is set, run:
```bash
echo $OPENAI_API_KEY
```

🏆 Great! You're all set to go. 

## Creating Customized Embeddings Using Your Dataset
Our framework supports creating customized embeddings using your own text corpus. Our framework only requires plain text documents for training without any need for labels. Some example use cases that you might want to train your own embeddings include:
- **Healthcare**: You have a dataset of patient notes and you want to create embeddings for each patient note to apply to a downstream task such as patient matching or clustering.
- **Customer Support**: You have a dataset of customer reviews for a hotel and you want to use the embeddings to understand the customer's preferences for different types of hotels.
- **News Recommendation**: You have a dataset of news articles and you want to create embeddings for each news article to apply to a downstream task such as news article recommendation or clustering.

To proceed, you need to prepare your dataset in the format of a JSON file containing a list of strings, where each string represents a document. For example:
```json
[
    "This is a training document",
    "This is another training document"
]
```
Assuming your dataset is preprocessed and stored in `data/dataset.json`, you can write your own training script to use our framework. Example training script:
```python
from cqg import ContrastiveQuestionGeneration
from mbqa import MBQA
import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with open("data/dataset.json", "r") as f:
        doc_texts = json.load(f)
    cqg = ContrastiveQuestionGeneration(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="DatasetName",
    )
    cqg.generate_questions()

    mbqa = MBQA(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="DatasetName",
    )
    mbqa.collect_training_data_with_cqg()
    mbqa.train_model()
```
Arguments for the `ContrastiveQuestionGeneration` (CQG) class:
- `corpus`: List of documents
- `temp_folder`: Temporary folder for storing intermediate files
- `name`: Name of the dataset

Arguments for the `MBQA` class:
- `corpus`: List of documents
- `temp_folder`: Temporary folder for storing intermediate files
- `output_folder`: Output folder for storing the trained model
- `name`: Name of the dataset

After running both the CQG and MBQA pipeline, you will have two files in the `./output/DatasetName` folder:
- `questions.json`: A JSON file containing the generated questions. This is needed for producing result interpretation when using the model for inference.
- `mbqa_model.pt`: A PyTorch model checkpoint. This is needed for loading the model for inference.

For more details on the CQG and MBQA pipeline, please refer to the [CQG](docs/CQG.md) and [MBQA](docs/MBQA.md) documentation.

🏆 Fantastic! You have successfully trained your own CQG-MBQA model checkpoint. Now let's proceed to use the model for inference in section [Using Pretrained Model Checkpoints](#using-pretrained-model-checkpoints).

## Using Pretrained Model Checkpoints
We provide the pre-trained model used in our experiments in the `checkpoints` folder. You can use the model to perform inference. To find the embedding quality of our pretrained model in various domains, we evaluate it on the [MTEB benchmark](https://huggingface.co/blog/mteb). The results are given in the [Evaluation on MTEB Benchmark](#evaluation-on-mteb-benchmark) section.

> 💡 If you have run through the [Creating Customized Embeddings Using Your Dataset](#creating-customized-embeddings-using-your-dataset) section, you should have a trained model checkpoint `mbqa_model.pt` and the generated questions `questions.json` similar to the ones in the `checkpoints/CQG-MBQA` folder. You may substitute the pre-trained model with your trained model in the following steps.

Our trained model can be easily used in a similar way to other sentence-transformer models. For example, to encode a list of documents:
```python
from mbqa_model import MultiTaskClassifier, MBQAMTEBModelWrapper
import torch
import json
import os

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "../checkpoints/CQG-MBQA/questions.json"), "r") as f: # change the path to your questions.json
    linear_questions = json.load(f)

model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1") # change the backbone to your trained model if it is different

model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/CQG-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0")) # change the path to your trained model checkpoint
    
model.to("cuda")

documents = [
    "This is a test document", 
    "This is another test document"
]
# Perform inference
embeddings = model.encode(documents) 
```
> 💡 The `MBQAMTEBModelWrapper` is wrapper class for using the trained model for inference. It can be used to evaluate the model on the MTEB benchmark or simply to get the embeddings for a list of documents.

> ⚠️ If you want to use your own trained model, please modify the path to your trained model checkpoint and questions.json file. And the actual backbone used in your MBQA training.

🏆 That's it! Try it out with your own downstream tasks!

## Reproducing Results in Our Paper
To ensure a reproducible pipeline, we provide a set of scripts and configurations tailored to our experiments. You may choose to run the experiments with our checkpoints (or train the model with our hyperparameters settings). To train the model with MEDI2 dataset, please follow all the steps below. To evaluate with our pre-trained models, please go straight to the [Running Evaluations](#running-evaluations) section.

The steps below to reproduce key results:
1. **Download and Prepare Data**: Download the MEDI2 dataset (70GB space required). 
2. **Run Preprocessing**: Preprocess text data to format supported by the framework.
3. **Train the Model**: Execute our training script.
4. **Evaluate the Model**: Evaluate the model on a diverse set of downstream tasks.

### Running the Pipeline with Default Configurations
#### MEDI2 Dataset
The MEDI2 dataset is available on [Hugging Face](https://huggingface.co/datasets/GritLM/MEDI2).

> ❗ At least 70GB of free space is required for downloading the dataset and saving the preprocessed files.

To use the dataset:
1. Clone the MEDI2 repository and pull LFS files
```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/GritLM/MEDI2
```
2. Run the preprocessing steps
```bash
cd ..
python framework/preprocess-medi2.py
```

#### Execute the Pipeline (Our Model)
```bash
python framework/run-cqg-mbqa-MEDI2.py
```

This creates an `output/MEDI2` folder in the working directory, containing:
- `questions.json`: Generated questions from the pipeline
- `mbqa_model.pt`: Model checkpoint trained on the MEDI2 dataset for CQG-MBQA

#### Execute the Pipeline (QA-Emb)
```bash
python framework/run-qaemb-mbqa-MEDI2.py
```

This creates an `output/MEDI2-QAEmb` folder in the working directory, containing:
- `questions.json`: Generated questions from the pipeline
- `mbqa_model.pt`: Model checkpoint trained on the MEDI2 dataset for QAEmb-MBQA

#### Use pre-trained model for inference
Alternatively, we provide the pre-trained model used in our experiments in the `checkpoints` folder. You can use the model to perform the evaluations.

### Running Evaluations

#### MTEB Benchmark Evaluation
To evaluate using the MTEB benchmark with cognitive load score included for the STS tasks, we need to install our modified benchmark package:
```bash
cd mteb
pip install .
```

Then modify the `run-mteb-cqg-mbqa.py` and `run-mteb-qaemb-mbqa.py` files to specify the tasks you want to evaluate on, and make sure the `questions.json` and `mbqa_model.pt` files paths are correct for CQG-MBQA or QAEmb-MBQA. 

The results will be saved to the `results_mteb_cqg` and `results_mteb_qaemb` folders.

#### NewSpectrum Dataset
Due to copyright issues, please contact the authors for the dataset or crawl it yourself.

To run evaluations on NewSpectrum dataset, modify the `framework/newspectrum.py` file make sure the `questions.json` and `mbqa_model.pt` files paths are correct for CQG-MBQA or QAEmb-MBQA. 

The script runs the evaluation on the NewSpectrum dataset for the models below and saves the results to the `results/newspectrum_scores.json` file. Models included in the evaluation are:
- [Sentence-BERT](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [OpenAI](https://platform.openai.com/docs/guides/embeddings)
- [GloVe](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d)
- [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- [Unsupervised SimCSE](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased)
- [Supervised SimCSE](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased)
- [Bag of Tokens](framework/utils.py)
- [CQG-MBQA](checkpoints/CQG-MBQA)
- [QAEmb-MBQA](checkpoints/QAEmb-MBQA)

#### MS MARCO Dataset
To run the (subset) MS MARCO dataset, modify the `framework/msmarco.py` file to specify the tasks you want to evaluate on, and make sure the `questions.json` and `mbqa_model.pt` files paths are correct for CQG-MBQA or QAEmb-MBQA. You can modify the `SAMPLING_RATIO` parameter to sample more or less documents from the MS MARCO dataset.

The script runs the evaluation on the (subset) MS MARCO dataset for the models similar to the ones in the [NewSpectrum Dataset](#newspectrum-dataset) and saves the results to the `results/msmarco_scores.json` file.


## Bugs and Issues
We will make a public accessible non-anonymous repository for the framework soon. If you have any questions, please raise an issue in this repository after deanonymization. 

## Citation
If you find this repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{
    anonymous2024a,
    title={A General Framework for Producing Interpretable Semantic Text Embeddings},
    author={Anonymous},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=23uY3FpQxc},
    note={under review}
}
```