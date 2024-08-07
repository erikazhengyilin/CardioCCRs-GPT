import json
import random
import pandas as pd
import argparse
from tqdm import tqdm

from client import create_client
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], required=True)
parser.add_argument('--sample_mode', type=str, choices=["random_k", "top_k", "zeroshot"], required=True)
parser.add_argument('--sample_size', type=int, required=True)
args = parser.parse_args()

def load_dataset_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def format_case(case):
    attrs_cols = [
        "Life Style",
        "Family History",
        "Social History",
        "Medical\/Surgical History",
        "Signs and Symptoms",
        "Comorbidities",
        "Diagnostic Techniques and Procedures",
        "Laboratory Values",
        "Pathology",
        "Pharmacological Therapy",
        "Interventional Therapy",
        "Age",
        "Gender"
    ]

    descriptions = "; ".join(f"{col}: {case.get(col, 'NA') if case.get(col) is not None else 'NA'}" for col in attrs_cols)
    diagnosis = case.get('Diagnosis', 'NA') if case.get('Diagnosis') is not None else 'NA'
    short_diagnosis = case.get('Short Diagnosis', 'NA') if case.get('Short Diagnosis') is not None else 'NA'

    return descriptions, diagnosis, short_diagnosis

def top_k_similar_case(case, train_dataset, k):
    case_embedding = np.array(case["case_embedding"]).reshape(1, -1)
    train_embeddings = np.array([train_case["case_embedding"] for train_case in train_dataset]).reshape(len(train_dataset), -1)
    similarities = cosine_similarity(case_embedding, train_embeddings).flatten()
    topk_idx = similarities.argsort()[-k:]
    topk = [train_dataset[idx] for idx in topk_idx]
    return topk

def create_fewshot_prompt(examples):
    prompt = ""
    for example in examples:
        descriptions, diagnosis, short_diagnosis = format_case(example)
        prompt += f"Input: {descriptions}\n Diagnosis: {short_diagnosis}\n\n"
    return prompt

def pred_diagnosis_fewshot(fewshot_prompt, case_description, client):
    prompt = f"Here are some examples,\n {fewshot_prompt} Now for Input: {case_description}\n, what is the diagnosis? \
        Just give me the top diagnosis, without any additional words or formatting."
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of predicting diagnoses \
             and short diagnoses based on detailed medical case descriptions."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content, response.usage.total_tokens 

def pred_diagnosis_zeroshot(case_description, client):
    prompt = f"For a clinical case like this: {case_description}\n, what is the diagnosis? \
        Just give me the top diagnosis, without any additional words." # zero shot
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of predicting diagnoses \
             and short diagnoses based on detailed medical case descriptions."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content, response.usage.total_tokens 

def main():
    train_dataset = load_dataset_json('dataset/cardio_maccrs_train_data.json')
    test_dataset = load_dataset_json('dataset/cardio_maccrs_test_data.json')
    client = create_client()

    results = []
    total_pred_token_usage = 0
    total_grade_token_usage = 0

    for i, case in tqdm(enumerate(test_dataset)):
        descriptions, _, diag_gt  = format_case(case)
        if args.sample_mode == 'random_k':
            k = args.sample_size
            train_sample = random.sample(train_dataset, k)
            fewshot_prompt = create_fewshot_prompt(train_sample)
            diag_pred, pred_token_usage = pred_diagnosis_fewshot(fewshot_prompt, descriptions, client)
        elif args.sample_mode == 'top_k':
            k = args.sample_size
            topk_cases = top_k_similar_case(case, train_dataset, k)
            fewshot_prompt = create_fewshot_prompt(topk_cases)
            diag_pred, pred_token_usage = pred_diagnosis_fewshot(fewshot_prompt, descriptions, client)
        elif args.sample_mode == 'zeroshot':
            diag_pred, pred_token_usage = pred_diagnosis_zeroshot(descriptions, client)

        cos_score = cal_cosine_similarity(diag_gt.lower(), diag_pred.lower(), client, model='small')
        grade, grade_token_usage = grade_diagnosis(diag_gt, diag_pred, client)

        total_pred_token_usage += pred_token_usage
        total_grade_token_usage += grade_token_usage
        results.append({"diag_gt": diag_gt.lower(), "diag_pred": diag_pred.lower(), "cos_score": cos_score, "gpt_grade": grade})

    print(f"Total tokens used for case prediction query: {total_pred_token_usage}")
    print(f"Total tokens used for grading query: {total_grade_token_usage}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{args.model}_{args.sample_mode}_{args.sample_size}_results.csv', index=False)

if __name__ == "__main__":
    main()