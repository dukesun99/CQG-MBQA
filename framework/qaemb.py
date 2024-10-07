import logging
import random
import string
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans
from enum import Enum
import pickle
from scipy.spatial.distance import cdist
import random
from openai import OpenAI

logger = logging.getLogger(__name__)
from utils import parse_response, format_qa_prompt, call_batch_api, wait_for_batch_api_completion, OpenAIWrapper

class QAEmbStep(Enum):
    NOT_STARTED = 0
    BATCH_JSON_CREATED = 1
    BATCH_REQUESTS_SENT = 2
    BATCH_REQUESTS_COMPLETED = 3
    RESULTS_RETRIEVED = 4
    QUESTIONS_EXTRACTED = 5
    QUESTIONS_PICKED = 6
    FINAL_QUESTIONS_GENERATED = 7

class QAEmbQuestionGeneration:
    def format_qaemb_prompt(self, positive_chunks):
        txt = '''Generate 10 diverse insightful yes/no questions that determine properties of an article.

**Reference Articles:** 
        '''
        i = 0
        for chunk in positive_chunks:
            txt += f"{i + 1}. {chunk}\n"
        i += 1
        txt += '''
**Example Questions:** 
        '''
        for i, question in enumerate(QAEmbQuestionGeneration.example_questions):
            txt += f"{i + 1}. {question}\n"
        txt+='''
**Instruction:** Based on the excerpts provided, generate 10 yes/no questions that can determine properties of the articles. Format the questions in a numbered list as shown below:
1. [First yes/no question]
2. [Second yes/no question]'''
    
        return txt
    
    example_questions = [
        "Is the sentence expressing skepticism or disbelief towards something or someone?",
        "Does the sentence include dialogue?",
        "Does the sentence describe a relationship between people? ",
        "Does the sentence involve the mention of a specific object or item?",
        "Does the sentence include technical or specialized terminology? ",
    ]
    
    def __init__(self, **kwargs):
        self.encoder = kwargs.get("encoder", "WhereIsAI/UAE-Large-V1")
        self.LLM = kwargs.get("LLM", "gpt-4o-mini")
        self.num_generations = kwargs.get("num_generations", 10)
        self.theta = kwargs.get("theta", 0.925)
        self.temp_folder = kwargs.get("temp_folder", "./temp")
        self.seed = kwargs.get("seed", 42)
        self.openai_api_key = kwargs.get("openai_api_key", None)
        self.device = kwargs.get("device", "cuda")
        self.name = kwargs.get("name", None) # used for resuming the process
        self.corpus = kwargs.get("corpus", None)
        assert self.corpus is not None, "Corpus is required"
        if self.name is None:
            # random name
            self.name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        self.temp_folder = os.path.join(self.temp_folder, self.name)
        os.makedirs(self.temp_folder, exist_ok=True)
        self.encoder_model = SentenceTransformer(self.encoder, device="cpu")
        if self.device is not None:
            self.encoder_model.to(self.device)
        self.encoder_model.eval()
        if self.openai_api_key is not None:
            self.client = OpenAIWrapper(self.openai_api_key)
        else:
            self.client = OpenAIWrapper().client
    def _log_progress(self, progress):
        with open(os.path.join(self.temp_folder, "progress.txt"), "w") as f:
            f.write(str(progress))
    
    def _get_progress(self):
        if not os.path.exists(os.path.join(self.temp_folder, "progress.txt")):
            return QAEmbStep.NOT_STARTED.value
        with open(os.path.join(self.temp_folder, "progress.txt"), "r") as f:
            return int(f.read())
    
    def generate_questions(self):
        corpus = self.corpus
        
        lock_file = os.path.join(self.temp_folder, "lock")
        if os.path.exists(lock_file):
            logger.error(f"Lock file {lock_file} exists, check if there is a process running... ")
            raise Exception(f"Lock file {lock_file} exists, check if there is a process running... ")
        with open(lock_file, "w") as f:
            f.write("locked")
        
        logger.info(f"Starting generation")
        
        if self._get_progress() < QAEmbStep.BATCH_JSON_CREATED.value:
            # 1. randomly construct samples of documents for the generation
            batch_questions_generation_jsons = []
            for i in range(self.num_generations):
                random_indices = random.sample(range(len(corpus)), self.seed)
                sampled_docs = [corpus[j][:10000] for j in random_indices]
                
                prompt = self.format_qaemb_prompt(sampled_docs)
                
                req_obj = {
                    "custom_id": f"{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.LLM,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                }
                
                batch_questions_generation_jsons.append(req_obj)
                
            # 2. save the batch json
            MAX_BATCH_SIZE = 40000
            for i in range(0, len(batch_questions_generation_jsons), MAX_BATCH_SIZE):
                batch = batch_questions_generation_jsons[i:i+MAX_BATCH_SIZE]
                with open(os.path.join(self.temp_folder, f"batch_questions_generation_{i}.json"), "w") as f:
                    for req in batch:
                        json.dump(req, f)
                        f.write("\n")
            
            self._log_progress(QAEmbStep.BATCH_JSON_CREATED.value)
            logger.info(f"Batch json created")
        
        if self._get_progress() < QAEmbStep.BATCH_REQUESTS_SENT.value:
            # 3. call the batch api
            json_files = os.listdir(self.temp_folder)
            json_files = [f for f in json_files if f.startswith("batch_questions_generation_")]
            json_files_full_path = [os.path.join(self.temp_folder, json_file) for json_file in json_files]
            
            call_batch_api(self.client, json_files_full_path, os.path.join(self.temp_folder, "batch_ids_questions_generation.json"))
            
            self._log_progress(QAEmbStep.BATCH_REQUESTS_SENT.value)
            logger.info(f"Batch API calls finished")
            
        if self._get_progress() < QAEmbStep.BATCH_REQUESTS_COMPLETED.value:
            # 4. wait for the batch API calls to complete
            with open(os.path.join(self.temp_folder, "batch_ids_questions_generation.json"), "r") as f:
                batch_ids_questions_generation = json.load(f)
            
            wait_for_batch_api_completion(self.client, batch_ids_questions_generation)
            
            self._log_progress(QAEmbStep.BATCH_REQUESTS_COMPLETED.value)
            logger.info(f"Batch API calls completed")
            
        if self._get_progress() < QAEmbStep.RESULTS_RETRIEVED.value:
            # 5. retrieve the results
            with open(os.path.join(self.temp_folder, "batch_ids_questions_generation.json"), "r") as f:
                batch_ids_questions_generation = json.load(f)
            
            for batch_id in batch_ids_questions_generation:
                batch = self.client.batches.retrieve(batch_id)
                batch_output_file_id = batch.output_file_id
                
                with open(os.path.join(self.temp_folder, f"output_batch_jsons_questions_generation_{batch_id}.json"), "wb") as f:
                    f.write(self.client.files.content(batch_output_file_id).content)
                    
            self._log_progress(QAEmbStep.RESULTS_RETRIEVED.value)
            logger.info(f"Batch API results retrieved")
        
        
        if self._get_progress() < QAEmbStep.QUESTIONS_EXTRACTED.value:
            questions = {}
            json_files = os.listdir(self.temp_folder)
            json_files = [f for f in json_files if f.startswith("output_batch_jsons_questions_generation_")]
            json_files_full_path = [os.path.join(self.temp_folder, json_file) for json_file in json_files]
            
            for json_file in json_files_full_path:
                with open(json_file, "r") as f:
                    for line in f:
                        response = json.loads(line)
                        custom_id = response["custom_id"]
                        questions[custom_id] = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])
            
            with open(os.path.join(self.temp_folder, "questions.json"), "w") as f:
                json.dump(questions, f)
            
            self._log_progress(QAEmbStep.QUESTIONS_EXTRACTED.value)
            logger.info(f"Questions extracted")
        
        if self._get_progress() < QAEmbStep.QUESTIONS_PICKED.value:
            deduped_questions = []
            picked_encoded_questions = []
            
            with open(os.path.join(self.temp_folder, "questions.json"), "r") as f:
                questions = json.load(f)
            
            for i in range(10):
                random_ids = list(range(len(questions)))
                random.shuffle(random_ids)
                
                for j in random_ids:
                    if i < len(questions[str(j)]):
                        question_text = questions[str(j)][i]
                        emb_q = self.encoder_model.encode([question_text], show_progress_bar=False)[0]
                        emb_q_norm = np.linalg.norm(emb_q)
                        emb_q = emb_q / emb_q_norm
                        is_good = True
                        for q in picked_encoded_questions:
                            if np.dot(emb_q, q) > self.theta:
                                is_good = False
                                break
                        if is_good:
                            deduped_questions.append(question_text)
                            picked_encoded_questions.append(emb_q)
                            
            with open(os.path.join(self.temp_folder, "deduped_questions.json"), "w") as f:
                json.dump(deduped_questions, f)
            
            self._log_progress(QAEmbStep.QUESTIONS_PICKED.value)
            logger.info(f"Questions picked")
                
                
        logger.info(f"Generation finished")
                
                
                