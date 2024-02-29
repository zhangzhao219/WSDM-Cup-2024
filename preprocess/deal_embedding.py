import os
import json
import numpy as np
from tqdm import tqdm

data_folder = "../embeddings"
body_dict_name = "gpt_ada2_4.npy"
body_file_name = "body_embeddings_20240107.jsonl"

if os.path.exists(os.path.join(data_folder, body_dict_name)):
    body_dict = np.load(
        os.path.join(data_folder, body_dict_name), allow_pickle=True
    ).item()
    print("body_dict:", len(body_dict))
else:
    body_dict = {}


def read_embedding_jsonl(body_file_name):
    f_body = open(os.path.join(data_folder, body_file_name), "r")
    body_data = f_body.readlines()
    f_body.close()

    body_temp_dict = {}
    for index, body in enumerate(tqdm(body_data)):
        body_str = json.loads(body)
        body_id = body_str["metadata"]
        body_embedding = np.array(body_str["data"][0]["embedding"])
        if len(body_embedding) != 1536:
            print(index, len(body_embedding))
            exit()
        body_temp_dict[body_id] = body_embedding

    for key in body_temp_dict.keys():
        body_dict[key] = body_temp_dict[key]


read_embedding_jsonl(body_file_name)
np.save(os.path.join(data_folder, body_dict_name), body_dict)
print("body_dict:", len(body_dict))
