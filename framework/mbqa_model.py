from torch import nn
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class MultiTaskClassifier(nn.Module):
    def __init__(self, num_labels=10000, backbone="WhereIsAI/UAE-Large-V1"):
        super(MultiTaskClassifier, self).__init__()
        self.transformer = SentenceTransformer(backbone, device="cpu")
        self.hidden_size = self.transformer.get_sentence_embedding_dimension()
        # Define an MLP for each classifier head
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, 8),  # First linear layer
                nn.ReLU(),                       # ReLU activation function
                nn.Linear(8, 1)                  # Output layer
            ) for _ in range(num_labels)
        ])
        
    def forward(self, texts, task_ids=None):
        embeddings = self.transformer.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        if task_ids is not None:
            logits = [self.classifiers[task_id](embeddings).squeeze(-1) for task_id in task_ids]
        else:
            logits = [classifier(embeddings) for classifier in self.classifiers]
        # Ensure all tensors in logits have at least one dimension
        logits = [l.unsqueeze(-1) if l.dim() == 0 else l for l in logits]
        logits = torch.cat(logits, dim=-1)
        logits = logits.squeeze(-1)
        # print(logits.shape)
        return logits

class MBQAMTEBModelWrapper():
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))   
    
    def __init__(self, model, questions, is_binary=False, binary_threshold=0.5, is_sparse=False, use_sigmoid=False):
        self.model = model
        self.is_binary = is_binary
        self.binary_threshold = binary_threshold
        self.is_sparse = is_sparse
        self.questions = questions
        self.use_sigmoid = use_sigmoid
            
    def encode(self, sentences: list[str], **kwargs):
        encoded = []
        batch_size = 64
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            logits = self.model(batch).cpu().detach().numpy()
            if self.use_sigmoid:
                logits = self.sigmoid(logits)
            if self.is_binary:
                binary_logits = np.array(logits > self.binary_threshold, dtype=np.float32)
                encoded.append(binary_logits)
            else:
                encoded.append(np.array(logits))
        stack_ed = np.vstack(encoded)
        return stack_ed
    
    def explain(self, embedding1, embedding2, num_explanations=None, verbose=False):
        # normalize the embeddings
        norm_embedding1 = embedding1 / np.linalg.norm(embedding1)
        norm_embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # compute the element-wise product
        product = norm_embedding1 * norm_embedding2
        
        absolute_product = np.abs(product)
        
        # count non-zero elements
        non_zero_count = np.count_nonzero(absolute_product)
        count = 0
        if verbose:
            for dim in range(len(product)):
                if product[dim] != 0:
                    print(f"Dimension {dim} Question: {self.questions[dim]}, Article1: {embedding1[dim]}, Article2: {embedding2[dim]}, Score: {product[dim]}")
                    count += 1
                    if num_explanations and count >= num_explanations:
                        break
        return non_zero_count