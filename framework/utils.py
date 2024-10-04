import re
import json
import time
from openai import OpenAI
import logging
import os

logger = logging.getLogger(__name__)

def parse_response(response):
    try:
        # Split the response into lines
        lines = response.split('\n')
        
        # Initialize an empty list to store questions
        questions = []
        
        # Loop through each line and extract the questions
        for line in lines:
            # Use regex to extract the question part
            match = re.match(r'\d+\.\s+(.*)', line)
            if match:
                questions.append(match.group(1))
        
        return questions
    except:
        return []
    
    
def format_qa_prompt(chunk, questions):
    txt = '''Evaluate the following text chunk based on the yes/no questions provided.

**Text Chunk:** '''
    txt += chunk + "\n\n"
    txt += '''**Questions:**
'''
    for i, question in enumerate(questions):
        txt += f"{i + 1}. {question}\n"
    
    txt += '''

**Instruction for the model:** Please read the provided text chunk and answer each of the questions with either "yes" or "no".Format the responses as follows:
1. yes/no
2. yes/no'''
        
    return txt

def call_batch_api(client, json_files, request_id_save_file_name):
    batch_ids = []
    for json_file in json_files:
        batch_file = client.files.create(
            file=open(json_file, 'rb'),
            purpose="batch"
        )
        batch_file_id = batch_file.id
        batch_request = client.batches.create(
            input_file_id = batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Batch request for file " + json_file
            }
        )
        batch_ids.append(batch_request.id)
    
    with open(request_id_save_file_name, "w") as f:
        json.dump(batch_ids, f)
        
        
def wait_for_batch_api_completion(client, batch_ids):
    for batch_id in batch_ids:
        while True:
            batch = client.batches.retrieve(batch_id)
            if batch.status == "completed":
                break
            time.sleep(60)
            

class OpenAIWrapper:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OpenAIWrapper, cls).__new__(cls)
            API_KEY = kwargs.get("API_KEY", None)
            if API_KEY is not None:
                os.environ["OPENAI_API_KEY"] = API_KEY
            cls._instance.client = OpenAI()
            try:
                test_response = cls._instance.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, how are you?"}
                    ]
                )
                logger.info(f"Connected to OpenAI")
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                logger.error(f"Error connecting to OpenAI: {e}\nNote we currently only support OpenAI models. Make sure you have setup the API key correctly or have passed the API key in as parameter. Make sure you have sufficient credits. Refer to https://platform.openai.com/docs/quickstart for more details on how to setup the API key.")
                raise e
        return cls._instance
    
    