from cqg import ContrastiveQuestionGeneration
from mbqa import MBQA
import subprocess
import json
import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Starting MEDI2")

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, "../data/medi2_documents.json"), "r") as f:
        doc_texts = json.load(f)
    doc_texts = doc_texts[:30000]
    logger.info(f"Loaded {len(doc_texts)} documents")
    
    test_mode = True
    
    if test_mode:
        if os.path.exists("/home/sunyq/CQG-MBQA/temp/MEDI2/lock"):
            # remove the lock file
            os.remove("/home/sunyq/CQG-MBQA/temp/MEDI2/lock")
        
    cqg = ContrastiveQuestionGeneration(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="MEDI2",
        k=100,
    )
    cqg.generate_questions()
    
    mbqa = MBQA(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="MEDI2",
    )
    mbqa.collect_training_data()
    mbqa.train_model()