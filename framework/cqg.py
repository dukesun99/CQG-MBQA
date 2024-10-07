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

class Step(Enum):
    NOT_STARTED = 0
    ENCODE_CORPUS = 1
    CLUSTER_CORPUS = 2
    SAMPLE_DOCUMENTS = 3
    BATCH_JSON_CREATED = 4
    BATCH_REQUESTS_SENT = 5
    BATCH_REQUESTS_COMPLETED = 6
    RESULTS_RETRIEVED = 7
    QUESTIONS_EXTRACTED = 8
    PROBE_BATCH_JSON_CREATED = 9
    PROBE_BATCH_REQUESTS_SENT = 10
    PROBE_BATCH_REQUESTS_COMPLETED = 11
    PROBE_RESULTS_RETRIEVED = 12
    PROBE_QUESTIONS_EXTRACTED = 13
    QUESTIONS_PICKED = 14
    FINAL_QUESTIONS_GENERATED = 15

    
class ContrastiveQuestionGeneration:
    def format_prompt(self,positive_chunks, negative_chunks):
        txt = '''Generate 10 simple yet insightful yes/no questions that determine properties of an article, where for all questions, the answer will be "yes" for ALL the positive articles and "no" for ALL the negative articles. Keep questions concise and avoid using complex sentence structures with "and" or "or" unless necessary.

**Positive Articles:** 
    '''
        i = 0
        for chunk in positive_chunks:
            txt += f"Positive {i + 1}. {chunk}\n"
            i += 1
    
        txt += '''
**Negative Articles:** 
    '''
        i = 0
        for chunk in negative_chunks:
            txt += f"Negative {i + 1}. {chunk}\n"
            i += 1
        txt += '''
**Instruction:** Based on the excerpts provided, generate 10 simple yet insightful yes/no questions that can accurately differentiate the positive articles from the negative articles. Each question should be concise and framed in such a way that it will elicit a "yes" response for ALL positive articles and a "no" response for ALL negative articles. Avoid using complex sentence structures with "and" or "or" unless absolutely necessary. Format the questions in a numbered list as shown below:
1. [First simple yes/no question]
2. [Second simple yes/no question]'''
        
        return txt
    
    def __init__(self, **kwargs):
        # parameters
        self.encoder = kwargs.get("encoder", "WhereIsAI/UAE-Large-V1")
        self.LLM = kwargs.get("LLM", "gpt-4o-mini")
        self.k = kwargs.get("k", 5000)
        self.n_pos = kwargs.get("n_pos", 6)
        self.n_hard_neg = kwargs.get("n_hard_neg", 18)
        self.n_easy_neg = kwargs.get("n_easy_neg", 18)
        self.p_pos = kwargs.get("p_pos", 5)
        self.p_hard_neg = kwargs.get("p_hard_neg", 3)
        self.p_easy_neg = kwargs.get("p_easy_neg", 2)
        assert self.p_pos + self.p_hard_neg + self.p_easy_neg <= 20, "The sum of p_pos, p_hard_neg and p_easy_neg should be less than or equal to 20"
        self.theta = kwargs.get("theta", 0.8)
        self.t = kwargs.get("t", 4)
        self.temp_folder = kwargs.get("temp_folder", "./temp")
        # self.output_folder = kwargs.get("output_folder", "./output")
        self.seed = kwargs.get("seed", 42)
        self.openai_api_key = kwargs.get("openai_api_key", None)
        self.device = kwargs.get("device", "cuda")
        self.name = kwargs.get("name", None) # used for resuming the process
        self.corpus = kwargs.get("corpus", None)
        assert self.corpus is not None, "Corpus is required"
        if self.name is None:
            # random name
            self.name = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        self.temp_folder = os.path.join(self.temp_folder, self.name)
        # self.output_folder = os.path.join(self.output_folder, self.name)
        os.makedirs(self.temp_folder, exist_ok=True)
        # os.makedirs(self.output_folder, exist_ok=True)
        self.encoder_model = SentenceTransformer(self.encoder, device="cpu")
        if self.device is not None:
            self.encoder_model.to(self.device)
        self.encoder_model.eval()
        if self.openai_api_key is not None:
            self.client = OpenAIWrapper(API_KEY=self.openai_api_key).client
        else:
            self.client = OpenAIWrapper().client
        
    def _log_progress(self, progress):
        with open(os.path.join(self.temp_folder, "progress.txt"), "w") as f:
            f.write(str(progress))

    def _get_progress(self):
        if not os.path.exists(os.path.join(self.temp_folder, "progress.txt")):
            return Step.NOT_STARTED.value
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
        
        # 1. encode the corpus
        if self._get_progress() < Step.ENCODE_CORPUS.value:
            logger.info(f"Encoding the corpus, this might take very long (several hours) depending on the size of the corpus and your GPU...")
            if self.encoder_model.device.type == "cpu":
                logger.warning(f"Encoding the corpus on CPU. This might take very long (several days)...")
            
            # 1.1 encode the corpus
            corpus_embeddings = self.encoder_model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

            # 1.2 normalize the embeddings
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

            # 1.3 cache the embeddings
            np.save(os.path.join(self.temp_folder, "corpus_embeddings.npy"), corpus_embeddings)
            
            self._log_progress(Step.ENCODE_CORPUS.value)
        else:
            corpus_embeddings = np.load(os.path.join(self.temp_folder, "corpus_embeddings.npy"))
        
        logger.info(f"Corpus encoded")
        
        # 2. Cluster the corpus
        if self._get_progress() < Step.CLUSTER_CORPUS.value:
            logger.info(f"Clustering the corpus...")
            kmeans = KMeans(n_clusters=self.k, random_state=self.seed).fit(corpus_embeddings)
            with open(os.path.join(self.temp_folder, "kmeans.pkl"), "wb") as f:
                pickle.dump(kmeans, f)
            self._log_progress(Step.CLUSTER_CORPUS.value)
        else:
            with open(os.path.join(self.temp_folder, "kmeans.pkl"), "rb") as f:
                kmeans = pickle.load(f)
        logger.info(f"Corpus clustered")
        
        cluster_assignments = kmeans.predict(corpus_embeddings)
        with open(os.path.join(self.temp_folder, "cluster_assignments.json"), "w") as f:
            json.dump(list([int(i) for i in cluster_assignments]), f)
            
        cluster_centers = []
        
        for i in range(self.k):
            cluster_indices = np.where(cluster_assignments == i)[0]
            cluster_embeddings = corpus_embeddings[cluster_indices]
            cluster_center = np.mean(cluster_embeddings, axis=0)
            cluster_centers.append(cluster_center)
        
        cluster_centers = np.array(cluster_centers)

        cluster_distances = cdist(cluster_centers, cluster_centers) 
        with open(os.path.join(self.temp_folder, "cluster_distances.npy"), "wb") as f:
            np.save(f, cluster_distances)

        # for each cluster, get the closest 3 clusters
        closest_clusters = np.argsort(cluster_distances, axis=1)[:, 1:4]
        
        # 3. stragtic sampling for positive and negative documents
        if self._get_progress() < Step.SAMPLE_DOCUMENTS.value:
            logger.info(f"Sampling documents...")
            initial_data = []
            
            for i in range(self.k):
                cluster_indices = np.where(cluster_assignments == i)[0]
                cluster_embeddings = corpus_embeddings[cluster_indices]
                cluster_center = cluster_centers[i]
                closest_indices = []
                for j in closest_clusters[i]:
                    closest_indices.extend(np.where(cluster_assignments == j)[0])
                closest_indices = list(set(closest_indices))
                
                if len(cluster_indices) < self.n_pos: 
                    positive_indices = cluster_indices
                    logger.warning(f"Cluster {i} size smaller than num_positives. Generally this should be fine, but if you see a lot of these warnings, please check if the number of clusters (k) is too high for your corpus.")
                else:
                    positive_indices = random.sample(list(cluster_indices), self.n_pos)
                
                if len(closest_indices) < self.n_hard_neg:
                    # sample all the closest indices
                    negative_indices = closest_indices
                    logger.warning(f"Cluster {i} size smaller than num_hard_negatives. Generally this should be fine, but if you see a lot of these warnings, please check if the number of clusters (k) is too high for your corpus.")
                else:
                    negative_indices = random.sample(closest_indices, self.n_hard_neg)
                
                
                for j in range(self.n_easy_neg):
                    random_ind = random.randint(0, len(corpus_embeddings) - 1)
                    while random_ind in cluster_indices or random_ind in closest_indices:
                        random_ind = random.randint(0, len(corpus_embeddings) - 1)
                        
                    negative_indices.append(random_ind)
                
                initial_data.append((positive_indices, negative_indices))

            with open(os.path.join(self.temp_folder, "initial_data.pkl"), "wb") as f:
                pickle.dump(initial_data, f)
            
            self._log_progress(Step.SAMPLE_DOCUMENTS.value)
        else:
            with open(os.path.join(self.temp_folder, "initial_data.pkl"), "rb") as f:
                initial_data = pickle.load(f)
        
        logger.info(f"Documents sampled")
        
        # 4. generate questions by calling the LLM
        
        # 4.1 construct the batch jsons
        logger.info(f"Preparing questions generation requests...")
        if self._get_progress() < Step.BATCH_JSON_CREATED.value:
            batch_questions_generation_jsons = []

            for i in range(len(initial_data)):
                positive_indices, negative_indices = initial_data[i]
                positive_chunks = []
                negative_chunks = []
                for j in positive_indices:
                    if len(corpus[j]) > 10000:
                        logger.warning(f"Document of index {j} is too long. Truncating to 10000 characters. This should be handled outside the framework. It is fine occassionally, but if you see this warning often, it is better to preprocess the corpus.")
                    positive_chunks.append(corpus[j][:10000]) # 10000 characters to safe guard the cost
                for j in negative_indices:
                    if len(corpus[j]) > 10000:
                        logger.warning(f"Document of index {j} is too long. Truncating to 10000 characters. This should be handled outside the framework. It is fine occassionally, but if you see this warning often, it is better to preprocess the corpus.")
                    negative_chunks.append(corpus[j][:10000])
                
                prompt = self.format_prompt(positive_chunks, negative_chunks)
                req_obj = {
                            "custom_id": f"{i}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                # "model": model_evaluation,
                                "model": self.LLM,
                                "messages": [{"role": "system", "content": prompt}]
                            }
                        }
                
                batch_questions_generation_jsons.append(req_obj)

            MAX_BATCH_SIZE = 40000
            
            for i in range(0, len(batch_questions_generation_jsons), MAX_BATCH_SIZE):
                batch = batch_questions_generation_jsons[i:i+MAX_BATCH_SIZE]
                with open(os.path.join(self.temp_folder, f"batch_questions_generation_jsons_{i}.json"), "w") as f:
                    for req in batch:
                        json.dump(req, f)
                        f.write("\n")
            
            self._log_progress(Step.BATCH_JSON_CREATED.value)
            logger.info(f"Questions generation requests prepared")
        
        # 4.2 call the batch API
        if self._get_progress() < Step.BATCH_REQUESTS_SENT.value:
            logger.info(f"Calling the batch API...")
            logger.info(f"Please DO NOT kill the process at this moment. It will take a while to complete... Progress will be lost if you kill the process and you will still be billed for the requests.")
            
            json_files = os.listdir(self.temp_folder)
            json_files = [f for f in json_files if f.startswith("batch_questions_generation_jsons_")]
            json_files_full_path = [os.path.join(self.temp_folder, json_file) for json_file in json_files]
            
            call_batch_api(self.client, json_files_full_path, os.path.join(self.temp_folder, "batch_ids_question_gen.json"))
                
            self._log_progress(Step.BATCH_REQUESTS_SENT.value)
            logger.info(f"Batch API calls finished")
        
        if self._get_progress() < Step.RESULTS_RETRIEVED.value:
            with open(os.path.join(self.temp_folder, "batch_ids_question_gen.json"), "r") as f:
                batch_ids_question_gen = json.load(f)
        # 4.3 wait for the batch API calls to complete
        if self._get_progress() < Step.BATCH_REQUESTS_COMPLETED.value:
            logger.info(f"Waiting for the batch API calls to complete...")
            wait_for_batch_api_completion(self.client, batch_ids_question_gen)
            self._log_progress(Step.BATCH_REQUESTS_COMPLETED.value)
            logger.info(f"Batch API calls completed")
        else:
            logger.info(f"Batch API calls already completed")
        
        # 4.4 get the results
        if self._get_progress() < Step.RESULTS_RETRIEVED.value:
            logger.info(f"Retrieving the results...")
            for batch_id in batch_ids_question_gen:
                batch = self.client.batches.retrieve(batch_id)
                batch_output_file_id = batch.output_file_id
                
                with open(os.path.join(self.temp_folder, f"output_batch_questions_generation_jsons_{batch_id}.json"), "wb") as f:
                    f.write(self.client.files.content(batch_output_file_id).content)
            
            self._log_progress(Step.RESULTS_RETRIEVED.value)
            logger.info(f"Results retrieved")
        else:
            logger.info(f"Results already retrieved")
        
        # 4.5 parse the results
        if self._get_progress() < Step.QUESTIONS_EXTRACTED.value:
            questions = {}
            
            output_files = os.listdir(self.temp_folder)
            output_files = [f for f in output_files if f.startswith("output_batch_questions_generation_jsons_")]
            
            for output_file in output_files:
                with open(os.path.join(self.temp_folder, output_file), "r") as f:
                    for line in f:
                        response = json.loads(line)
                        custom_id = int(response["custom_id"])
                        questions[custom_id] = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])

            with open(os.path.join(self.temp_folder, "questions.json"), "w") as f:
                json.dump(questions, f)
            
            self._log_progress(Step.QUESTIONS_EXTRACTED.value)
            logger.info(f"Questions extracted")
        
        else:
            with open(os.path.join(self.temp_folder, "questions.json"), "r") as f:
                questions = json.load(f)
        
            logger.info(f"Questions already extracted")
            
        # 5. Probe the quality of the questions
        # 5.1 create the batch jsons
        if self._get_progress() < Step.PROBE_BATCH_JSON_CREATED.value:
            logger.info(f"Creating probe batch json...")
            contexts_initial_filtering = []
            batch_jsons_initial_filtering = []
            
            for i in range(self.k):
                cluster_indices = np.where(cluster_assignments == i)[0]
                cluster_embeddings = corpus_embeddings[cluster_indices]
                cluster_center = cluster_centers[i]
                closest_indices = []
                for j in closest_clusters[i]:
                    closest_indices.extend(np.where(cluster_assignments == j)[0])
                closest_indices = list(set(closest_indices))
                
                if len(cluster_indices) < self.p_pos:
                    positive_indices = cluster_indices
                    logger.warning(f"Cluster {i} size smaller than num_positives. Generally this should be fine, but if you see a lot of these warnings, please check if the number of clusters (k) is too high for your corpus.")
                else:
                    positive_indices = random.sample(list(cluster_indices), self.p_pos)
                
                if len(closest_indices) < self.p_hard_neg:
                    negative_indices = closest_indices
                    logger.warning(f"Cluster {i} size smaller than num_hard_negatives. Generally this should be fine, but if you see a lot of these warnings, please check if the number of clusters (k) is too high for your corpus.")
                else:
                    negative_indices = random.sample(closest_indices, self.p_hard_neg)
                    
                
                for j in range(self.p_easy_neg):
                    random_ind = random.randint(0, len(corpus_embeddings) - 1)
                    retry_count = 0
                    while random_ind in cluster_indices or random_ind in closest_indices:
                        random_ind = random.randint(0, len(corpus_embeddings) - 1)
                        retry_count += 1
                        if retry_count > 100:
                            logger.warning(f"Failed to find a random index for cluster {i}. This should be fine, but if you see this warning often, please check if the number of clusters (k) is too high for your corpus.")
                            break
                    if retry_count > 100:
                        break
                    negative_indices.append(random_ind)
                    
                generated_questions = questions[i]
                
                context = {
                    "positive": [corpus[j][:10000] for j in positive_indices],
                    "negative": [corpus[j][:10000] for j in negative_indices],
                }
                
                contexts_initial_filtering.append(context)
                
                for j in range(len(positive_indices)):
                    prompt = format_qa_prompt(corpus[positive_indices[j]], generated_questions)
                    req_obj = {
                        "custom_id": f"{i}_{j}_positive",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.LLM,
                            "messages": [{"role": "system", "content": prompt}]
                        }
                    }
                    batch_jsons_initial_filtering.append(req_obj)
                    
                for j in range(len(negative_indices)):
                    prompt = format_qa_prompt(corpus[negative_indices[j]], generated_questions)
                    req_obj = {
                        "custom_id": f"{i}_{j}_negative",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.LLM,
                            "messages": [{"role": "system", "content": prompt}]
                        }
                    }
                    batch_jsons_initial_filtering.append(req_obj)
            
            for i in range(0, len(batch_jsons_initial_filtering), MAX_BATCH_SIZE):
                batch = batch_jsons_initial_filtering[i:i+MAX_BATCH_SIZE]
                with open(os.path.join(self.temp_folder, f"batch_jsons_initial_filtering_{i}.json"), "w") as f:
                    for req in batch:
                        json.dump(req, f)
                        f.write("\n")
            
            self._log_progress(Step.PROBE_BATCH_JSON_CREATED.value)
            logger.info(f"Probe batch json created")
        
        # 5.2 call the batch API
        if self._get_progress() < Step.PROBE_BATCH_REQUESTS_SENT.value:
            logger.info(f"Calling the batch API...")
            logger.info(f"Please DO NOT kill the process at this moment. It will take a while to complete... Progress will be lost if you kill the process and you will still be billed for the requests.")
            
            json_files = os.listdir(self.temp_folder)
            json_files = [f for f in json_files if f.startswith("batch_jsons_initial_filtering_")]
            json_files_full_path = [os.path.join(self.temp_folder, json_file) for json_file in json_files]
            
            call_batch_api(self.client, json_files_full_path, os.path.join(self.temp_folder, "batch_ids_initial_filtering.json"))
            
            self._log_progress(Step.PROBE_BATCH_REQUESTS_SENT.value)
            logger.info(f"Batch API calls finished")
        
        if self._get_progress() < Step.PROBE_RESULTS_RETRIEVED.value:
            with open(os.path.join(self.temp_folder, "batch_ids_initial_filtering.json"), "r") as f:
                batch_ids_initial_filtering = json.load(f)
        
        # 5.3 wait for the batch API calls to complete
        if self._get_progress() < Step.PROBE_BATCH_REQUESTS_COMPLETED.value:
            logger.info(f"Waiting for the batch API calls to complete...")
            wait_for_batch_api_completion(self.client, batch_ids_initial_filtering)
            self._log_progress(Step.PROBE_BATCH_REQUESTS_COMPLETED.value)
            logger.info(f"Batch API calls completed")
        else:
            logger.info(f"Batch API calls already completed")
            
        # 5.4 get the results
        if self._get_progress() < Step.PROBE_RESULTS_RETRIEVED.value:
            logger.info(f"Retrieving the results...")
            for batch_id in batch_ids_initial_filtering:
                batch = self.client.batches.retrieve(batch_id)
                batch_output_file_id = batch.output_file_id
                
                with open(os.path.join(self.temp_folder, f"output_batch_jsons_initial_filtering_{batch_id}.json"), "wb") as f:
                    f.write(self.client.files.content(batch_output_file_id).content)
                    
            self._log_progress(Step.PROBE_RESULTS_RETRIEVED.value)
            logger.info(f"Results retrieved")
        else:
            logger.info(f"Results already retrieved")
            
        # 5.5 parse the results
        if self._get_progress() < Step.PROBE_QUESTIONS_EXTRACTED.value:
            logger.info(f"Parsing the results...")
            initial_filtering_responses = {}
            
            files_initial_filtering = os.listdir(self.temp_folder)
            files_initial_filtering = [f for f in files_initial_filtering if f.startswith("output_batch_jsons_initial_filtering_")]
            
            for file in files_initial_filtering:
                with open(os.path.join(self.temp_folder, file), "r") as f:
                    for line in f:
                        response = json.loads(line)
                        custom_id = response["custom_id"]
                        i, j, label = custom_id.split("_")

                        if i not in initial_filtering_responses:
                            initial_filtering_responses[i] = {
                                "positives": {},
                                "negatives": {}
                            }
                            
                        if label == "positive":
                            initial_filtering_responses[i]["positives"][j] = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])
                        else:
                            initial_filtering_responses[i]["negatives"][j] = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])
            
            with open(os.path.join(self.temp_folder, "initial_filtering_responses.json"), "w") as f:
                json.dump(initial_filtering_responses, f)
            
            self._log_progress(Step.PROBE_QUESTIONS_EXTRACTED.value)
            logger.info(f"Results parsed")
        
        else:
            with open(os.path.join(self.temp_folder, "initial_filtering_responses.json"), "r") as f:
                initial_filtering_responses = json.load(f)

        # 6. pick the best questions
        if self._get_progress() < Step.QUESTIONS_PICKED.value:
            logger.info(f"Picking the best questions...")
            picked_questions = {}
            all_question_scores = {}
            
            for i in range(len(initial_filtering_responses)):
                questions_results = {}
                
                for j in initial_filtering_responses[str(i)]["positives"]:
                    qas = initial_filtering_responses[str(i)]["positives"][str(j)]
                    for k in range(len(qas)):
                        if k not in questions_results:
                            questions_results[k] = {
                            "yes_in_positives": 0,
                            "yes_in_negatives": 0,
                            "no_in_positives": 0,
                            "no_in_negatives": 0,
                        }
                        if "yes" in qas[k]:
                            questions_results[k]["yes_in_positives"] += 1
                        else:
                            questions_results[k]["no_in_positives"] += 1
                            
                for j in initial_filtering_responses[str(i)]["negatives"]:
                    qas = initial_filtering_responses[str(i)]["negatives"][str(j)]
                    for k in range(len(qas)):
                        if k not in questions_results:
                            logger.warning(f"Question {k} not found in questions_results. This should not happen. This could due to the parsing error. Please raise an issue.")
                            continue
                        if "yes" in qas[k]:
                            questions_results[k]["yes_in_negatives"] += 1
                        else:
                            questions_results[k]["no_in_negatives"] += 1
                
                # score each question by % yes in positives - % yes in negatives
                
                question_scores = []
                for k in questions_results:
                    yes_in_positives = questions_results[k]["yes_in_positives"]
                    yes_in_negatives = questions_results[k]["yes_in_negatives"]
                    no_in_positives = questions_results[k]["no_in_positives"]
                    no_in_negatives = questions_results[k]["no_in_negatives"]

                    score = yes_in_positives / (yes_in_positives + no_in_positives) - yes_in_negatives / (yes_in_negatives + no_in_negatives)
                    question_scores.append((k, score))
                    
                question_scores.sort(key=lambda x: -x[1])
                all_question_scores[str(i)] = question_scores
                picked_questions[str(i)] = []
                encoded_picked_questions = []
                for k, _ in question_scores:
                    
                    emb_q = self.encoder_model.encode([questions[str(i)][k]], show_progress_bar=False)[0]
                    emb_q_norm = np.linalg.norm(emb_q)
                    emb_q = emb_q / emb_q_norm
                    is_good = True
                    for q in range(len(encoded_picked_questions)):
                        if np.dot(emb_q, encoded_picked_questions[q]) > self.theta:
                            is_good = False
                            logger.info(f"Question {k} is not good, question: {questions[str(i)][k]}")
                            break
                    if is_good:
                        encoded_picked_questions.append(emb_q)
                        picked_questions[str(i)].append(k)
                    if len(picked_questions[str(i)]) == self.t:
                        break
                
                if len(picked_questions[str(i)]) < self.t:
                    # add more questions
                    num_to_add = self.t - len(picked_questions[str(i)])
                    for k, _ in question_scores:
                        if k not in picked_questions[str(i)]:
                            picked_questions[str(i)].append(k)
                            num_to_add -= 1
                        if num_to_add == 0:
                            break
                
            with open(os.path.join(self.temp_folder, "picked_questions.json"), "w") as f:
                json.dump(picked_questions, f)
            
            self._log_progress(Step.QUESTIONS_PICKED.value)
            logger.info(f"Questions picked")
            
        else:
            with open(os.path.join(self.temp_folder, "picked_questions.json"), "r") as f:
                picked_questions = json.load(f)
        
        # 7. generate the final questions
        if self._get_progress() < Step.FINAL_QUESTIONS_GENERATED.value:
            logger.info(f"Generating the final questions...")
            
            deduped_questions = {}
            picked_encoded_questions = []
            
            for i in range(self.t):
                logger.info(f"Generating the final questions round {i}...")
                clusters_ids = list(range(len(picked_questions)))
                # randomly permute the cluster ids
                random.shuffle(clusters_ids)
                for j in clusters_ids:
                    if str(j) not in deduped_questions:
                        deduped_questions[str(j)] = []
                    question_text = questions[str(j)][picked_questions[str(j)][i]]
                    emb_q = self.encoder_model.encode([question_text], show_progress_bar=False)[0]
                    emb_q_norm = np.linalg.norm(emb_q)
                    emb_q = emb_q / emb_q_norm
                    is_good = True
                    for q in range(len(picked_encoded_questions)):
                        if np.dot(emb_q, picked_encoded_questions[q]) > self.theta:
                            is_good = False
                            break
                    if is_good:
                        picked_encoded_questions.append(emb_q)
                        deduped_questions[str(j)].append(picked_questions[str(j)][i])
                        
            total_deduped_questions = 0

            for i in deduped_questions:
                total_deduped_questions += len(deduped_questions[i])
        
            logger.info(f"Total deduped questions: {total_deduped_questions}")
            with open(os.path.join(self.temp_folder, "deduped_questions.json"), "w") as f:
                json.dump(deduped_questions, f)

            self._log_progress(Step.FINAL_QUESTIONS_GENERATED.value)
            logger.info(f"Final questions generated")
        # remove lock
        os.remove(os.path.join(self.temp_folder, "lock"))
    
    