from mbqa_model import MultiTaskClassifier
from utils import OpenAIWrapper, format_qa_prompt, call_batch_api, wait_for_batch_api_completion, parse_response
from enum import Enum
import os
import numpy as np
import random
import json
import os
import pickle
import logging
import torch
import time
from torch.optim import Adam

from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, task_ids, labels):
        self.texts = texts
        self.task_ids = task_ids
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'task_ids': torch.tensor(self.task_ids[idx]),  # List of task IDs relevant for this sample
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)  # List of labels for each task
        }

class StepMBQA(Enum):
    NOT_STARTED = 0
    REQUEST_JSON_CREATED = 1
    REQUEST_SENT = 2
    REQUEST_COMPLETED = 3
    RESULTS_RETRIEVED = 4
    FINAL_QUESTIONS_ARTICLE_PAIRS_GENERATED = 5
    
class MBQA:
    def __init__(self, **kwargs):
        self.corpus = kwargs.get("corpus", None)
        self.temp_folder = kwargs.get("temp_folder", "./temp")
        self.output_folder = kwargs.get("output_folder", "./output")
        self.name = kwargs.get("name", None)
        assert self.name is not None, "Name is required"
        self.temp_folder = os.path.join(self.temp_folder, self.name)
        self.output_folder = os.path.join(self.output_folder, self.name)
        
        self.LLM = kwargs.get("LLM", "gpt-4o-mini")
        self.openai_api_key = kwargs.get("openai_api_key", None)
        self.client = OpenAIWrapper(API_KEY=self.openai_api_key).client
        self.device = kwargs.get("device", "cuda")
        self.backbone = kwargs.get("backbone", "WhereIsAI/UAE-Large-V1")
        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.num_steps = kwargs.get("num_steps", 3000000)
        
    def _log_progress(self, progress):
        with open(os.path.join(self.temp_folder, "mbqa_progress.txt"), "w") as f:
            f.write(str(progress))

    def _get_progress(self):
        if not os.path.exists(os.path.join(self.temp_folder, "mbqa_progress.txt")):
            return StepMBQA.NOT_STARTED.value
        with open(os.path.join(self.temp_folder, "mbqa_progress.txt"), "r") as f:
            return int(f.read())
        
    def collect_training_data(self):
        with open(os.path.join(self.temp_folder, "deduped_questions.json"), "r") as f:
            deduped_questions = json.load(f)
            
        with open(os.path.join(self.temp_folder, "questions.json"), "r") as f:
            questions = json.load(f)
        
        with open(os.path.join(self.temp_folder, "cluster_assignments.json"), "r") as f:
            cluster_assignments = json.load(f)
            cluster_assignments = np.array(cluster_assignments)
        
        with open(os.path.join(self.temp_folder, "cluster_distances.npy"), "rb") as f:
            cluster_distances = np.load(f)
        
        logger.info(f"Start collecting training data...")
        
        # 1. Sample questions from each cluster and pack them into multiple of 20 to save cost
        if self._get_progress() < StepMBQA.REQUEST_JSON_CREATED.value:
            logger.info("Sampling questions from each cluster... ")
            linear_questions = []
            original_questions_id = []

            selected_docs_in_clusters = {}

            selected_positive_articles = {}
            docs_frequency = {}

            for i in range(len(deduped_questions)):
                if str(i) not in deduped_questions:
                    continue
                picked_questions_cluster = deduped_questions[str(i)]
                generated_questions_cluster = questions[str(i)]
                picked_questions_cluster_text = [generated_questions_cluster[q] for q in picked_questions_cluster]
                
                cluster_indices = np.where(cluster_assignments == i)[0]
                if len(cluster_indices) > 100:
                    cluster_indices = random.sample(list(cluster_indices), 100)
                else:
                    cluster_indices = list(cluster_indices)
                selected_docs_in_clusters[str(i)] = cluster_indices
                for j in range(len(picked_questions_cluster)):
                    linear_questions.append(picked_questions_cluster_text[j])
                    original_questions_id.append((i, picked_questions_cluster[j]))

                    question_id = len(linear_questions) - 1
                    
                    selected_positive_articles[question_id] = []
                    
                    for k in cluster_indices:
                        selected_positive_articles[question_id].append(k)
                        if k not in docs_frequency:
                            docs_frequency[k] = 0
                        docs_frequency[k] += 1
            
            selected_hard_negatives = {}
            closest_clusters_top5 = np.argsort(cluster_distances, axis=1)[:, 1:6]
            
            for q_id in range(len(original_questions_id)):
                i = original_questions_id[q_id][0]
                
                cluster_indices = selected_docs_in_clusters[str(i)]
                
                # get nearest 5 clusters of this cluster
                closest_clusters_of_i = closest_clusters_top5[i]
                
                if q_id not in selected_hard_negatives:
                    selected_hard_negatives[q_id] = []
                
                for j in range(len(closest_clusters_of_i)):
                    for k in selected_docs_in_clusters[str(closest_clusters_of_i[j])]:
                        selected_hard_negatives[q_id].append(k)
                        if k not in docs_frequency:
                            docs_frequency[k] = 0
                        docs_frequency[k] += 1
            
            selected_random_negatives = {}
            candidate_docs = set()
            for doc_index in docs_frequency:
                # if doc frequency is not a multiple of 20, add it to the candidate docs
                if docs_frequency[doc_index] % 20 != 0:
                    candidate_docs.add(doc_index)
                    
            selected_hard_negatives = {k: set(selected_hard_negatives[k]) for k in selected_hard_negatives}
            selected_positive_articles = {k: set(selected_positive_articles[k]) for k in selected_positive_articles}
            
            for q_id in range(len(original_questions_id)):
                i = original_questions_id[q_id][0]
                
                selected_random_negatives[q_id] = set()
                
                good_candidates = set(candidate_docs) - set(selected_random_negatives[q_id]) - set(selected_positive_articles[q_id]) - set(selected_hard_negatives[q_id])
                good_candidates_list = list(good_candidates)
                
                target_num_random_negatives = 1000 - len(selected_hard_negatives[q_id]) - len(selected_positive_articles[q_id])
                
                if len(good_candidates) < target_num_random_negatives:
                    random_doc_ind_good = good_candidates_list
                else:
                    random_doc_ind_good = random.sample(good_candidates_list, target_num_random_negatives)
                
                selected_random_negatives[q_id].update(random_doc_ind_good)
                
                for k in random_doc_ind_good:
                    docs_frequency[k] += 1
                    if docs_frequency[k] % 20 == 0:
                        candidate_docs.remove(k)
                
                assert len(selected_random_negatives[q_id]) + len(selected_positive_articles[q_id]) + len(selected_hard_negatives[q_id]) <= 1000
                
                while True:
                    random_ind = random.randint(0, len(self.corpus) - 1)
                    if random_ind not in docs_frequency:
                        docs_frequency[random_ind] = 1
                        selected_random_negatives[q_id].add(random_ind)
                        candidate_docs.add(random_ind)
                    if len(selected_random_negatives[q_id]) + len(selected_positive_articles[q_id]) + len(selected_hard_negatives[q_id]) >= 1000:
                        break
            # reorganize the dic to be doc_index -> list of question_id

            docs_to_question_ids = {}

            for q_id in range(len(original_questions_id)):
                for doc_index in selected_positive_articles[q_id]:
                    if doc_index not in docs_to_question_ids:
                        docs_to_question_ids[doc_index] = []
                    docs_to_question_ids[doc_index].append(q_id)
                    
                for doc_index in selected_hard_negatives[q_id]:
                    if doc_index not in docs_to_question_ids:
                        docs_to_question_ids[doc_index] = []
                    docs_to_question_ids[doc_index].append(q_id)
                    
                for doc_index in selected_random_negatives[q_id]:
                    if doc_index not in docs_to_question_ids:
                        docs_to_question_ids[doc_index] = []
                    docs_to_question_ids[doc_index].append(q_id)
                    
            len(docs_to_question_ids)

            # count the frequency of each doc in the selected docs that are not multiple of 20

            num_non_20 = 0

            for doc_index in docs_to_question_ids:
                if len(docs_to_question_ids[doc_index]) % 20 != 0:
                    num_non_20 += 1
                    
            logger.info(f"Number of documents that are failed to pack to multiple of 20: {num_non_20}")
            
            docs_to_question_ids_int = {int(k): [int(vi) for vi in v] for k, v in docs_to_question_ids.items()}
            
            with open(os.path.join(self.temp_folder, "docs_to_question_ids.json"), "w") as f:
                json.dump(docs_to_question_ids_int, f)
                
            with open(os.path.join(self.output_folder, "linear_questions.json"), "w") as f:
                json.dump(linear_questions, f)
            
            with open(os.path.join(self.temp_folder, "original_questions_id.json"), "w") as f:
                json.dump(original_questions_id, f)
            
            batch_jsons_get_training_data = []

            for doc_index in docs_to_question_ids:
                doc_text = self.corpus[doc_index][:10000]
                
                questions_ids_for_doc = docs_to_question_ids[doc_index]
                
                questions_texts = [linear_questions[q_id] for q_id in questions_ids_for_doc]
                
                batch_size = 20
                
                for i in range(0, len(questions_texts), batch_size):
                    batch = questions_texts[i:i+batch_size]
                    prompt = format_qa_prompt(doc_text, batch)
                    req_obj = {
                            "custom_id": f"{doc_index}_{i}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.LLM,
                                "messages": [{"role": "system", "content": prompt}]
                            }
                        }
                    batch_jsons_get_training_data.append(req_obj)
                    
            logger.info(f"Number of requests to get training data: {len(batch_jsons_get_training_data)}")
            
            MAX_BATCH_SIZE = 40000
            
            for i in range(0, len(batch_jsons_get_training_data), MAX_BATCH_SIZE):
                batch = batch_jsons_get_training_data[i:i+MAX_BATCH_SIZE]
                with open(os.path.join(self.temp_folder, f"batch_jsons_get_training_data_{i}.json"), "w") as f:
                    for req in batch:
                        json.dump(req, f)
                        f.write("\n")
            
            self._log_progress(StepMBQA.REQUEST_JSON_CREATED.value)
            logger.info(f"Requests to get training data prepared")
            
        
        
        if self._get_progress() < StepMBQA.REQUEST_SENT.value:
            # 2. call the batch API
            logger.info("Calling the batch API to get training data...")
            
            json_files = os.listdir(self.temp_folder)
            json_files = [f for f in json_files if f.startswith("batch_jsons_get_training_data_")]
            json_files = [os.path.join(self.temp_folder, f) for f in json_files]
            
            call_batch_api(self.client, json_files, os.path.join(self.temp_folder, "batch_ids_get_training_data.json"))

            self._log_progress(StepMBQA.REQUEST_SENT.value)
            logger.info(f"Requests to get training data sent")
            
        if self._get_progress() < StepMBQA.REQUEST_COMPLETED.value:
            # 3. wait for the batch API to complete
            logger.info("Waiting for the batch API to complete...")
            
            batch_ids = json.load(open(os.path.join(self.temp_folder, "batch_ids_get_training_data.json"), "r"))
            
            wait_for_batch_api_completion(self.client, batch_ids)
            
            self._log_progress(StepMBQA.REQUEST_COMPLETED.value)
            logger.info(f"Requests to get training data completed")
            
        if self._get_progress() < StepMBQA.RESULTS_RETRIEVED.value:
            # 4. retrieve the results
            logger.info("Retrieving the results...")
            
            batch_ids = json.load(open(os.path.join(self.temp_folder, "batch_ids_get_training_data.json"), "r"))
            
            for batch_id in batch_ids:
                batch = self.client.batches.retrieve(batch_id)
                batch_output_file_id = batch.output_file_id
                
                with open(os.path.join(self.temp_folder, f"batch_results_get_training_data_{batch_id}.json"), "wb") as f:
                    f.write(self.client.files.content(batch_output_file_id).content)
                    
                
            self._log_progress(StepMBQA.RESULTS_RETRIEVED.value)
            logger.info(f"Results retrieved")
            
        if self._get_progress() < StepMBQA.FINAL_QUESTIONS_ARTICLE_PAIRS_GENERATED.value:
            # 5. generate final questions article pairs
            logger.info("Generating final questions article pairs...")
            
            with open(os.path.join(self.temp_folder, "docs_to_question_ids.json"), "r") as f:
                docs_to_question_ids = json.load(f)
                
            with open(os.path.join(self.output_folder, "linear_questions.json"), "r") as f:
                linear_questions = json.load(f)
                
            with open(os.path.join(self.temp_folder, "original_questions_id.json"), "r") as f:
                original_questions_id = json.load(f)
                
            result_files = os.listdir(self.temp_folder)
            result_files = [f for f in result_files if f.startswith("batch_results_get_training_data_")]
            
            training_data = {}
            total_number_of_warnings = 0
            for result_file in result_files:
                for line in open(os.path.join(self.temp_folder, result_file), "r"):
                    response = json.loads(line)
                    custom_id = response["custom_id"]
                    doc_index, start_ind = custom_id.split("_")
                    # doc_index = int(doc_index)
                    start_ind = int(start_ind)
                    req_answers = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])
                    if len(req_answers) + start_ind > len(docs_to_question_ids[doc_index]):
                        logger.warning(f"Warning: number of answers not equal to number of questions. This error is handled but you should not see a lot of it.")
                        total_number_of_warnings += 1
                        continue
                    if doc_index not in training_data:
                        training_data[doc_index] = {}

                    for i in range(len(req_answers)):
                        training_data[doc_index][docs_to_question_ids[doc_index][start_ind + i]] = req_answers[i]
                
            with open(os.path.join(self.output_folder, "training_data.json"), "w") as f:
                json.dump(training_data, f)
                
            self._log_progress(StepMBQA.FINAL_QUESTIONS_ARTICLE_PAIRS_GENERATED.value)
            logger.info(f"Final questions article pairs generated")
        
        logger.info(f"Finished collecting training data")
        
    def train_model(self):
        logger.info(f"Start training model...")
        with open(os.path.join(self.output_folder, "training_data.json"), "r") as f:
            training_data = json.load(f)
        
        with open(os.path.join(self.output_folder, "linear_questions.json"), "r") as f:
            linear_questions = json.load(f)
        
        training_texts = []
        training_task_ids = []
        training_labels = []

        for doc_id in training_data:
            task_ids = list(training_data[doc_id].keys())
            task_labels = []
            for q_id in task_ids:
                task_labels.append(training_data[doc_id][q_id])
                
            training_texts.append(self.corpus[int(doc_id)])
            training_task_ids.append([int(q_id) for q_id in task_ids])
            training_labels.append(task_labels)


        if type(training_labels[0][0]) == str:
            training_labels = [[1 if "yes" in label else 0 for label in labels]for labels in training_labels]
            
        train_texts, val_texts, train_task_ids, val_task_ids, train_labels, val_labels = train_test_split(training_texts, training_task_ids, training_labels, test_size=0.1, random_state=42)
        
        train_dataset = MyDataset(
            train_texts,
            train_task_ids,
            train_labels
        )

        val_dataset = MyDataset(
            val_texts,
            val_task_ids,
            val_labels
        )
        
        data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        label_freq = {
            0: 0,
            1: 0
        }

        for labels in training_labels:
            for label in labels:
                label_freq[label] += 1
                
        weight = torch.tensor([label_freq[0] / label_freq[1]]).to(self.device)
        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone=self.backbone)
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weight)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        
        logger.info(f"Model training in progress...")
        num_steps = 0
        while True:
            if num_steps >= self.num_steps:
                break
            model.train()
            total_loss = 0
            losses = []
            t0 = time.time()
            for batch in data_loader:
                text = batch['text'][0]  # Assuming batch size of 1 for simplicity
                task_ids = batch['task_ids'][0]
                labels = batch['labels'][0]
                
                labels = labels.to(self.device)

                optimizer.zero_grad()
                if len(task_ids) < 2:
                    continue
                logits = model([text], task_ids=task_ids)

                # Calculate loss only for the active tasks
                loss = 0
                # print(logits.shape)
                # print(labels.shape)
                if logits.shape != labels.shape:
                    print("warning: logits shape is not equal to labels shape")
                    print(logits.shape, labels.shape)
                    continue
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_steps += 1
                
                if num_steps % 10000 == 0:
                    print(f"Average loss: {total_loss / 10000}", f"Time elapsed: {time.time() - t0}")
                    t0 = time.time()
                    losses.append(total_loss / 10000)
                    total_loss = 0
            
            model.eval()

            pred_labels = []
            gt_labels = []

            for batch in val_data_loader:
                text = batch['text'][0]  # Assuming batch size of 1 for simplicity
                task_ids = batch['task_ids'][0]
                labels = batch['labels'][0]
                
                labels = labels.to(self.device)
                if len(task_ids) == 0:
                    continue
                logits = model(text, task_ids=task_ids)
                pred_labels.append(logits.cpu().detach().numpy())
                gt_labels.append(labels.cpu().detach().numpy())
                
            pred_labels = np.concatenate(pred_labels, axis=0)
            gt_labels = np.concatenate(gt_labels, axis=0)

            logger.info(classification_report(gt_labels, pred_labels > 0))
            
        torch.save(model.state_dict(), os.path.join(self.output_folder, f"new_multi_task_classifier_uae_{num_steps}.pt"))    
        
            
        