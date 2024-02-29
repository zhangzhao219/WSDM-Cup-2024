import os
import json
import torch
import itertools
import pandas as pd
from rouge import Rouge
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


def lcs(s, t):
    len1 = len(s)
    len2 = len(t)
    res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if t[i - 1] == s[j - 1]:
                res[i][j] = 1 + res[i - 1][j - 1]
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    l = res[-1][-1]
    p = l / len1
    r = l / len2
    f = 2 * p * r / (p + r)
    return p, r, f


def generate_json(t, level, type_t, metric, score):
    t["level"] = level
    t["type"] = type_t
    t["metric"] = metric
    t["score"] = score
    return t


def calculate(data_list, test_data_temp, calculate_names):

    rouge = Rouge()
    sentence_transformer_model = SentenceTransformer(
        "../pretrained/nomic-ai/nomic-embed-text-v1", trust_remote_code=True
    ).eval()

    final_score_data_temp = []
    for index, data in enumerate(tqdm(test_data_temp)):
        torch.cuda.empty_cache()
        temp_json = {}

        temp_json["uuid"] = data["uuid"]
        # print(temp_json["uuid"])

        document_all = ""
        for document in data["documents"]:
            document_all = document_all + document + "\n"

        history_all = ""
        for i in data["history"]:
            history_all = history_all + i["question"] + "\n"
            history_all = history_all + i["answer"] + "\n"

        history_all = history_all + data["question"] + "\n"

        history_document_all = history_all + document_all

        document_all_emb = sentence_transformer_model.encode(document_all).reshape(
            1, -1
        )
        history_document_all_emb = sentence_transformer_model.encode(
            history_document_all
        ).reshape(1, -1)

        emb_list = []

        for i, file_data in enumerate(data_list):

            now_emb = sentence_transformer_model.encode(file_data[index]).reshape(1, -1)
            emb_list.append(now_emb)
            score = cosine_similarity(now_emb, document_all_emb)[0][0]
            write_json = generate_json(
                temp_json.copy(), "emb", "d_a_" + str(i), "s", score
            )
            final_score_data_temp.append(write_json)

            score = cosine_similarity(now_emb, history_document_all_emb)[0][0]
            write_json = generate_json(
                temp_json.copy(), "emb", "hqd_a_" + str(i), "s", score
            )
            final_score_data_temp.append(write_json)

            try:
                score = rouge.get_scores(
                    hyps=[file_data[index]], refs=[document_all], avg=True
                )["rouge-l"]
                p, r, f = score["p"], score["r"], score["f"]
            except:
                p, r, f = -2, -2, -2
            write_json = generate_json(
                temp_json.copy(), "word", "d_a_" + str(i), "p", p
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "word", "d_a_" + str(i), "r", r
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "word", "d_a_" + str(i), "f", f
            )
            final_score_data_temp.append(write_json)

            try:
                p, r, f = lcs(file_data[index].lower(), document_all.lower())
            except:
                p, r, f = -2, -2, -2
            write_json = generate_json(
                temp_json.copy(), "char", "d_a_" + str(i), "p", p
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "char", "d_a_" + str(i), "r", r
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "char", "d_a_" + str(i), "f", f
            )
            final_score_data_temp.append(write_json)

            try:
                score = rouge.get_scores(
                    hyps=[file_data[index]], refs=[history_document_all], avg=True
                )["rouge-l"]
                p, r, f = score["p"], score["r"], score["f"]
            except:
                p, r, f = -2, -2, -2
            write_json = generate_json(
                temp_json.copy(), "word", "hqd_a_" + str(i), "p", p
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "word", "hqd_a_" + str(i), "r", r
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "word", "hqd_a_" + str(i), "f", f
            )
            final_score_data_temp.append(write_json)

            try:
                p, r, f = lcs(file_data[index].lower(), history_document_all.lower())
            except:
                p, r, f = -2, -2, -2
            write_json = generate_json(
                temp_json.copy(), "char", "hqd_a_" + str(i), "p", p
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "char", "hqd_a_" + str(i), "r", r
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "char", "hqd_a_" + str(i), "f", f
            )
            final_score_data_temp.append(write_json)

        for pair in calculate_names:
            p0, p1 = pair[0], pair[1]

            score = cosine_similarity(emb_list[p0], emb_list[p1])[0][0]
            write_json = generate_json(
                temp_json.copy(), "emb", "a_" + str(p0) + "_a_" + str(p1), "s", score
            )
            final_score_data_temp.append(write_json)

            try:
                score = rouge.get_scores(
                    hyps=[data_list[p0][index]], refs=[data_list[p1][index]], avg=True
                )["rouge-l"]
                p, r, f = score["p"], score["r"], score["f"]

            except:
                p, r, f = -2, -2, -2

            write_json = generate_json(
                temp_json.copy(), "word", "a_" + str(p0) + "_a_" + str(p1), "p", p
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "word", "a_" + str(p0) + "_a_" + str(p1), "r", r
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "word", "a_" + str(p0) + "_a_" + str(p1), "f", f
            )
            final_score_data_temp.append(write_json)

            try:
                p, r, f = lcs(
                    data_list[p0][index].lower(), data_list[p1][index].lower()
                )
            except:
                p, r, f = -2, -2, -2

            write_json = generate_json(
                temp_json.copy(), "char", "a_" + str(p0) + "_a_" + str(p1), "p", p
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "char", "a_" + str(p0) + "_a_" + str(p1), "r", r
            )
            final_score_data_temp.append(write_json)
            write_json = generate_json(
                temp_json.copy(), "char", "a_" + str(p0) + "_a_" + str(p1), "f", f
            )
            final_score_data_temp.append(write_json)
    return final_score_data_temp


if __name__ == "__main__":

    data_folder = "data/test/20240212"

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

    with open("../data/wsdm/prepare/ori/release_phase2_test_data_wo_gt.json", "r") as f:
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
