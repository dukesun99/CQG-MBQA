from cqg import ContrastiveQuestionGeneration
from mbqa import MBQA
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
    
    logger.info(f"Loaded {len(doc_texts)} documents")
        
    cqg = ContrastiveQuestionGeneration(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="MEDI2",
    )
    cqg.generate_questions()
    
    mbqa = MBQA(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="MEDI2",
    )
    mbqa.collect_training_data_with_cqg()
    mbqa.train_model()