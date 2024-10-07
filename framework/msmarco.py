from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import pathlib, os
import logging
import random
from utils import BagOfTokenEncoder
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import bm25s
from mbqa_model import MultiTaskClassifier, MBQAMTEBModelWrapper

SAMPLING_RATIO = 0.01

class SBERTEncodingModelGeneral:
    def __init__(self, model, device="cuda"):
        self.model = SentenceTransformer(model, device=device)
        
    def encode(self, sentences, **kwargs):
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False, **kwargs)

class OpenAIEncodingModelGeneralWrapper:
    def __init__(self, model):
        self.model = model
        self.client = OpenAI()
        
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        embeddings = []
        batch_size = 1000
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding batches"):
            response = self.client.embeddings.create(input=sentences[i:i+batch_size], model=self.model)
            embeddings_batch = [item.embedding for item in response.data]
            embeddings.extend(embeddings_batch)
        return np.array(embeddings)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dirname = os.path.dirname(__file__)
def get_model(model_name):
    if model_name == "CQG-MBQA":
        with open(os.path.join(dirname, "../checkpoints/CQG-MBQA/questions.json"), "r") as f:
            linear_questions = json.load(f)

        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")
        model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/CQG-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
        model.to("cuda")
        mteb_model = MBQAMTEBModelWrapper(model, linear_questions, is_binary=True, is_sparse=False, binary_threshold=0.5, use_sigmoid=True)
    
    elif model_name == "QAEmb-MBQA":
        with open(os.path.join(dirname, "../checkpoints/QAEmb-MBQA/questions.json"), "r") as f:
            linear_questions = json.load(f)
        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")
        model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/QAEmb-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
        model.to("cuda")
        mteb_model = MBQAMTEBModelWrapper(model, linear_questions, is_binary=True, is_sparse=False, binary_threshold=0.5, use_sigmoid=True)
    elif model_name == "bm25":
        mteb_model = None
    elif model_name == "bot":
        mteb_model = BagOfTokenEncoder()
    elif model_name == "openai":
        mteb_model = OpenAIEncodingModelGeneralWrapper('text-embedding-3-large')
    else:
        mteb_model = SBERTEncodingModelGeneral(model_name)
    
    return mteb_model

dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(dirname, "../data/BEIR")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

logger.info(f"Loaded {len(corpus)} documents")

gt_docs = set()
for k in qrels:
    for d in qrels[k]:
        gt_docs.add(d)
len(gt_docs)

# subsample 1% of the corpus, but keep all the gt_docs
# Set a seed for reproducibility
random.seed(42)

selected_corpus = set(gt_docs)  # Start with all ground truth documents

# Calculate how many additional documents we need to reach 1% of the corpus
target_size = int(len(corpus) * SAMPLING_RATIO) # Change this to test a subset only
additional_docs_needed = max(0, target_size - len(selected_corpus))

# Randomly select additional documents from the corpus
remaining_docs = list(set(corpus.keys()) - selected_corpus)  # Convert to list
additional_docs = random.sample(remaining_docs, min(additional_docs_needed, len(remaining_docs)))

selected_corpus.update(additional_docs)

# Create a new corpus dictionary with only the selected documents
subsampled_corpus = {doc_id: corpus[doc_id] for doc_id in selected_corpus}

logger.info(f"Original corpus size: {len(corpus)}")
logger.info(f"Subsampled corpus size: {len(subsampled_corpus)}")
logger.info(f"Percentage of original corpus: {len(subsampled_corpus) / len(corpus) * 100:.2f}%")
logger.info(f"Number of ground truth documents: {len(gt_docs)}")
logger.info(f"All ground truth documents included: {set(gt_docs).issubset(set(subsampled_corpus.keys()))}")

def run_evaluation(model):
    # Extract query IDs and texts while maintaining order
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    # Extract document IDs and texts while maintaining order
    doc_ids = list(subsampled_corpus.keys())
    doc_texts = [subsampled_corpus[doc_id]['text'] for doc_id in doc_ids]

    if model is not None:
        # Encode queries
        query_encodings = model.encode(query_texts)
        # Encode documents
        doc_encodings = model.encode(doc_texts)
        # 2. Normalize Encodings for Cosine Similarity
        query_encodings = normalize(query_encodings)
        doc_encodings = normalize(doc_encodings)
    else:
        corpus_tokens = bm25s.tokenize(doc_texts)
        bm25_retriever = bm25s.BM25()
        bm25_retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(query_texts)
        all_doc_ids, all_scores = bm25_retriever.retrieve(query_tokens, k=10)
 
    # 3. Compute Cosine Similarity and Evaluate NDCG@10
    ndcg_scores = []

    if model is not None:
        # Iterate over queries using their indices to maintain mapping
        for idx in tqdm(range(len(query_ids)), desc="Processing Queries"):
            query_id = query_ids[idx]
            query_encoding = query_encodings[idx].reshape(1, -1)  # Reshape for cosine_similarity

            # Compute cosine similarities between the current query and all documents
            similarities = np.dot(query_encoding, doc_encodings.T)[0]

            # Get top 10 document indices based on similarity
            top_indices = np.argsort(similarities)[::-1][:10]
            
            # Retrieve the actual document IDs for the top indices
            top_doc_ids = [doc_ids[i] for i in top_indices]
            
            # Get relevance scores for the top 10 documents
            relevances = [
                qrels.get(query_id, {}).get(doc_id, 0.0) for doc_id in top_doc_ids
            ]
            
            # Compute NDCG@10 for the current query using sklearn's ndcg_score
            ndcg = ndcg_score([relevances], [similarities[top_indices]], k=10)
            ndcg_scores.append(ndcg)
    else:
        for idx in tqdm(range(len(query_ids)), desc="Processing Queries"):
            query_id = query_ids[idx]
            retrieved_doc_ids = [doc_ids[i] for i in all_doc_ids[idx]]
            relevances = [qrels.get(query_id, {}).get(doc_id, 0) for doc_id in retrieved_doc_ids[:10]]
            similarities = [all_scores[idx][i] for i in range(10)]
            
            ndcg = ndcg_score([relevances], [similarities], k=10)
            ndcg_scores.append(ndcg)
            
    average_ndcg = np.mean(ndcg_scores)
    return average_ndcg

models = [
    "openai",
    "bm25",
    "bot",
    'sentence-transformers/average_word_embeddings_glove.6B.300d',
    'google-bert/bert-base-uncased',
    'princeton-nlp/unsup-simcse-bert-base-uncased',
    'princeton-nlp/sup-simcse-bert-base-uncased',
    'sentence-transformers/all-MiniLM-L12-v2',
    "CQG-MBQA",
    "QAEmb-MBQA",
]

scores = {}
for model in models:
    mteb_model = get_model(model)
    ndcg = run_evaluation(mteb_model)
    print(f"{model}: {ndcg}")
    scores[model] = ndcg
    
with open(os.path.join(dirname, "../results/msmarco_scores.json"), "w") as f:
    json.dump(scores, f)
    