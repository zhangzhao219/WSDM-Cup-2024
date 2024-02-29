import os
import json
import torch
import itertools
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def evenly_divide_list(lst, lst2, num_groups):
    n = len(lst)
    q, r = divmod(n, num_groups)
    group_sizes = [q] * num_groups
    for i in range(r):
        group_sizes[i] += 1

    groups = []
    lst2_groups = []
    i = 0
    for size in group_sizes:
        group = lst[i : i + size]
        lst2_g = []
        for k in lst2:
            lst2_g.append(k[i : i + size])
        lst2_groups.append(lst2_g)
        groups.append(group)
        i += size
    return groups, lst2_groups


def generate_json(t, level, type_t, metric, score):
    t["level"] = level
    t["type"] = type_t
    t["metric"] = metric
    t["score"] = score
    return t


def calculate(data_list, test_data_temp, calculate_names):

    sentence_transformer_model = SentenceTransformer(
        "pretrained/nomic-ai/nomic-embed-text-v1", trust_remote_code=True
    ).eval()

    final_score_data_temp = []
    for index, data in enumerate(tqdm(test_data_temp)):
        torch.cuda.empty_cache()
        temp_json = {}

        temp_json["uuid"] = data["uuid"]

        emb_list = []

        for file_data in data_list:

            now_emb = sentence_transformer_model.encode(file_data[index]).reshape(1, -1)
            emb_list.append(now_emb)


        for pair in calculate_names:
            p0, p1 = pair[0], pair[1]

            score = cosine_similarity(emb_list[p0], emb_list[p1])[0][0]
            write_json = generate_json(
                temp_json.copy(), "emb", "a_" + str(p0) + "_a_" + str(p1), "s", score
            )
            final_score_data_temp.append(write_json)

    return final_score_data_temp


if __name__ == "__main__":

    data_folder = "merge"

    data_list = []

    for file in sorted(os.listdir(data_folder)):
        if not file.endswith(".jsonl"):
            continue
        f = open(os.path.join(data_folder, file),"r")
        data = f.readlines()
        result_data = []
        for i in data:
            result_data.append(json.loads(i)["response"])
        data_list.append(result_data)
        f.close()

    calculate_names = list(
        itertools.combinations([i for i in range(len(data_list))], 2)
    )

    with open("data/release_phase2_test_data_wo_gt.json", "r") as f:
        test_data = json.load(f)

    num_group = 16
    test_data_list, temp_data_list = evenly_divide_list(test_data, data_list, num_group)

    final_score_data_p = []
    ctx = torch.multiprocessing.get_context("spawn")
    p = ctx.Pool(num_group)
    for i in range(0, num_group):
        final_score_data_p.append(
            p.apply_async(
                calculate,
                args=(
                    temp_data_list[i],
                    test_data_list[i],
                    calculate_names
                ),
            )
        )
    p.close()
    p.join()
    final_score_data = []
    for i in final_score_data_p:
        final_score_data.extend(i.get())

    df = pd.json_normalize(final_score_data)
    df.to_csv(os.path.join(data_folder, "score.csv"), index=None)
