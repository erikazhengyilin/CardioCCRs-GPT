import pandas as pd
import numpy as np
import json
import os
import threading
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from client import create_client, get_embedding_small

def convert_xlsx_to_json(input_filepath, output_filepath):
    df = pd.read_excel(input_filepath)
    df.to_json(output_filepath, orient='records', indent=2)

def get_shortdiag_maccrs(diagnosis, client):
    prompt = f"Given the description: \"{diagnosis}\", provide the name of the disease the patient has in a short phrase. \
        If there are multiple diseases, separate them using \";\". \
        Prioritize cardio-related diseases by listing them first, with the major finding at the top. \
        For example, if the input is \"beneficial effect of sodium dichloroacetate in muscle cytochrome c oxidase deficiency\", \
        you will return \"muscle cytochrome c oxidase deficiency\". \
        If you are unable to find a diagnosis because the description does not contain any, return NA."

    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of extracting disease names from medical descriptions."},
            {"role": "user", "content": prompt}
        ],
    )
    # print(f"Total tokens used: {response.usage.total_tokens}")
    # print(f"Prompt tokens used: {response.usage.prompt_tokens}")
    # print(f"Completion tokens used: {response.usage.completion_tokens}")
    return response.choices[0].message.content.strip(), response.usage.total_tokens

def get_icd10_maccrs(diagnosis, client):
    prompt = f"Given the following list of diagnoses \"{diagnosis}\", provide the corresponding ICD-10 codes for each diagnosis, separated by \";\". \
        Just return the list of codes without any additional text. If entry is NA, or if you are unable to find an ICD-10 code, return NA. \
        For example, if the input is \"Ebstein's anomaly of tricuspid valve; rheumatic mitral stenosis\", \
        you will return \"Q22.5; I05.0\"."

    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in medical coding, tasked with mapping diagnoses to ICD-10 codes."},
            {"role": "user", "content": prompt}
        ],
    )
    # print(f"Total tokens used: {response.usage.total_tokens}")
    # print(f"Prompt tokens used: {response.usage.prompt_tokens}")
    # print(f"Completion tokens used: {response.usage.completion_tokens}")
    return response.choices[0].message.content.strip(), response.usage.total_tokens 

def process_maccrs_df(maccrs_dataset, client):
    # get shorter ver. of diagnosis from chatgpt
    total_token_usage = 0
    for item in maccrs_dataset:
        diagnosis = item["Diagnosis"]
        short_diag, token_usage = get_shortdiag_maccrs(diagnosis, client)
        total_token_usage += token_usage
        item["Short Diagnosis"] = short_diag
    print(f"Total tokens used for short diagnosis query: {total_token_usage}")

    # get icd10 code from chatgpt (mainly for train/test split)
    total_token_usage = 0
    for item in maccrs_dataset:
        diagnosis = item["Short Diagnosis"]
        icd10s, token_usage = get_icd10_maccrs(diagnosis, client)
        total_token_usage += token_usage
        item["ICD-10"] = icd10s
    print(f"Total tokens used for ICD-10 query: {total_token_usage}")

    return maccrs_dataset

def get_topicd10(filenames, output_filename):
    def formate_icd10_codes(codes):
        codes_list = [code.strip() for code in codes.replace(';', ',').split(',')]
        return ';'.join(codes_list)

    def select_top_icd10(codes):
        icd10_list = codes.split(';')
        icd10_cats = [code.split('.')[0].strip() for code in icd10_list]
        icd10_dict = OrderedDict()
        for cat in icd10_cats:
            if cat in icd10_dict:
                icd10_dict[cat] += 1
            else:
                icd10_dict[cat] = 1
                
        max_count = max(icd10_dict.values())
        for cat, count in icd10_dict.items():
            if count == max_count:
                return cat
        
    jsons_list = []
    for filename in filenames:
        df = pd.read_json(filename)
        jsons_list.append(df)

    file_lengths = [len(df) for df in jsons_list]
    if len(set(file_lengths)) != 1:
        raise ValueError("All JSON files must have the same number of records.")
    
    combined_json = jsons_list[0].copy()
    combined_json['ICD-10'] = combined_json['ICD-10'].apply(formate_icd10_codes)

    for i in range(1, len(jsons_list)):
        for j in range(len(combined_json)):
            # just double checking the df aligns every 10 record
            if j%10 == 0:
                if combined_json['CR Number'][j] != jsons_list[i]['CR Number'][j]:
                    print(f"Mismatch at index {j} in DataFrame {i}")
        icd10_list = jsons_list[i]['ICD-10'].apply(formate_icd10_codes)
        combined_json['ICD-10'] += ';' + icd10_list

    combined_json['ICD-10'] = combined_json['ICD-10'].apply(formate_icd10_codes)
    top_icd10_list = combined_json['ICD-10'].apply(select_top_icd10).tolist()
    combined_json['Top_ICD-10'] = top_icd10_list
    combined_json.to_json(output_filename, orient='records', indent=2)

def get_embeddings(filename, client):
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
        
    df = pd.read_json(filename)
    store_cols = [col for col in attrs_cols if col in df.columns]
    df['combined_text'] = df[store_cols].apply(lambda x: ' '.join(x.fillna('').astype(str)), axis=1)
    df['case_embedding'] = df['combined_text'].apply(
        lambda x: get_embedding_small(x, client).tolist() if x.strip() else None
    )
    df.drop(columns=['combined_text'], inplace=True)
    df.to_json(filename, orient='records', indent=2)

def main():
    # convert to json
    input_filepath = 'dataset/cardio_maccrs.xlsx'
    output_basepath = 'dataset/cardio_maccrs'
    client = create_client()
    output_filenames = []

    def process_and_save(i):
            output_filepath = f'{output_basepath}_{i}.json'
            output_filenames.append(output_filepath)
            convert_xlsx_to_json(input_filepath, output_filepath)

            with open(output_filepath, 'r') as file:
                maccrs_dataset = json.load(file)

            maccrs_dataset = process_maccrs_df(maccrs_dataset, client)

            with open(output_filepath, 'w') as file:
                json.dump(maccrs_dataset, file, indent=2)

    # query the short diagnosis and icd10 3 times, parallel (number of times is adjustable)
    threads = []
    for i in range(1, 4):
        t = threading.Thread(target=process_and_save, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    get_topicd10(output_filenames, output_basepath+'.json')
    get_embeddings(output_basepath+'.json', client)

    # remove temp. files
    for filename in output_filenames:
        os.remove(filename)

    # train test split
    df = pd.read_json(output_basepath+'.json')
    icd10_occurrence_counts = df['Top_ICD-10'].value_counts()
    df['Top_ICD-10'] = df['Top_ICD-10'].apply(lambda x: x if icd10_occurrence_counts[x] >= 2 else 'unique')
    X = df.drop(columns=['Top_ICD-10']) 
    y = df['Top_ICD-10']
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.9, stratify=y)
    X_train.to_json('dataset/cardio_maccrs_train_data.json', orient='records', indent=2)
    X_test.to_json('dataset/cardio_maccrs_test_data.json', orient='records', indent=2)

if __name__ == "__main__":
    main()

