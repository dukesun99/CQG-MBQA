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

import json

with open(os.path.join(dirname, "../data/50k_sbert_balanced_candidates.json"), "r", encoding="utf-8") as f:
    candidates = json.load(f)

titles = [candidate["title"] for candidate in candidates]
articles = [candidate["text"][:10000] for candidate in candidates]

num_samples = min(1000, len(titles))
np.random.seed(42)
sampled_indices = np.random.choice(len(titles), size=num_samples, replace=False)
queries = [titles[i] for i in sampled_indices]

def run_evaluation(model):
    if model is not None:
        # 1. Encode queries and articles
        encoded_queries = model.encode(queries)
        encoded_articles = model.encode(articles)
        # 2. Normalize Encodings for Cosine Similarity
        encoded_queries = normalize(encoded_queries)
        encoded_articles = normalize(encoded_articles)
    else:
        query_tokens = bm25s.tokenize(queries)
        corpus_tokens = bm25s.tokenize(articles)
        bm25_retriever = bm25s.BM25()
        bm25_retriever.index(corpus_tokens)
        all_doc_ids, all_scores = bm25_retriever.retrieve(query_tokens, k=10)
 
    # 3. Compute Cosine Similarity and Evaluate NDCG@10
    ndcg_scores = []

    if model is not None:
        # Iterate over queries using their indices to maintain mapping
        for idx in tqdm(range(len(queries)), desc="Processing Queries"):
            query_encoding = encoded_queries[idx].reshape(1, -1)  # Reshape for cosine_similarity

            # Compute cosine similarities between the current query and all documents
            similarities = np.dot(query_encoding, encoded_articles.T)[0]

            # Get top 10 document indices based on similarity
            top_indices = np.argsort(similarities)[::-1][:10]
            
            # Get relevance scores for the top 10 documents
            relevances = [1 if d_id == sampled_indices[idx] else 0 for d_id in top_indices]
            
            # Compute NDCG@10 for the current query using sklearn's ndcg_score
            ndcg = ndcg_score([relevances], [similarities[top_indices]], k=10)
            ndcg_scores.append(ndcg)
    else:
        for idx in tqdm(range(len(queries)), desc="Processing Queries"):
            retrieved_doc_ids = [i for i in all_doc_ids[idx]]
            relevances = [1 if d_id == sampled_indices[idx] else 0 for d_id in retrieved_doc_ids[:10]]
            similarities = [all_scores[idx][i] for i in range(10)]
            
            ndcg = ndcg_score([relevances], [similarities], k=10)
            ndcg_scores.append(ndcg)
            
    average_ndcg = np.mean(ndcg_scores)
    return average_ndcg

models = [
    "bm25",
    "bot",
    "openai",
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
    
with open(os.path.join(dirname, "../results/newspectrum_scores.json"), "w") as f:
    json.dump(scores, f)
    