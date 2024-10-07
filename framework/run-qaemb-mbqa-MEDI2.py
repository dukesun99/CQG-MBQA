from qaemb import QAEmbQuestionGeneration
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
        
    qaemb = QAEmbQuestionGeneration(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="MEDI2-QAEmb",
        num_generations=100
    )
    qaemb.generate_questions()
    
    mbqa = MBQA(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="MEDI2-QAEmb",
    )
    mbqa.collect_training_data_with_qaemb()
    mbqa.train_model()