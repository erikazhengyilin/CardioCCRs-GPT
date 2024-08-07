from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

chatgpt_api_path = 'api_keys/chatgpt.txt'

def read_api_key(filepath):
    with open(filepath, 'r') as file:
        return file.read().strip()
    
def create_client(model='chatgpt4o'):
    if model == 'chatgpt4o':
        api_key = read_api_key(chatgpt_api_path)
        client = OpenAI(api_key=api_key)
    if model == 'bert':
        client = SentenceTransformer('bert-base-nli-mean-tokens')
    return client

def get_embedding_ada(text, chatgpt_client):
    response = chatgpt_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

def get_embedding_small(text, chatgpt_client):
    response = chatgpt_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

def get_embedding_large(text, chatgpt_client):
    response = chatgpt_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding).reshape(1, -1)


def get_embedding_bert(text, bert_model):
    return bert_model.encode([text])