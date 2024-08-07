from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from client import *

def cal_cosine_similarity(text1, text2, client, model='small'):
    if model == 'adav2':
        cos_similarity = cosine_similarity(get_embedding_ada(text1, client), \
                                           get_embedding_ada(text2, client))[0][0]
    elif model == 'small':
        cos_similarity = cosine_similarity(get_embedding_small(text1, client), \
                                           get_embedding_small(text2, client))[0][0]
    elif model == 'large':
        cos_similarity = cosine_similarity(get_embedding_large(text1, client), \
                                           get_embedding_large(text2, client))[0][0]
    elif model == 'bert':
        cos_similarity = cosine_similarity(get_embedding_bert(text1, client), \
                                           get_embedding_bert(text2, client))[0][0]
    return cos_similarity

def cal_similarities(diag_list1, diag_list2, client, model='small'):
    results = []
    for i, diag1 in enumerate(diag_list1):
        diag2 = diag_list2[i]
        cos_similarity = cal_cosine_similarity(diag1, diag2, client, model)
        results.append(cos_similarity)
    return results, 

def grade_diagnosis(text1, text2, client):
    prompt = f"Please evaluate the accuracy of the predicted diagnosis '{text1}' compared to the ground truth diagnosis \
        '{text2}' on a scale of 'Inaccurate', 'Partially Accurate', and 'Accurate'. Please just return either one of these 3 phrases"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical assistant capable of assessing the accuracy of medical diagnoses. \
             Provide an evaluation based on the given ground truth and predicted diagnoses."},
            {"role": "user", "content": prompt}
        ],
    )
    grade = response.choices[0].message.content
    return grade, response.usage.total_tokens