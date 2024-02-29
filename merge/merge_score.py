import os
import json
import zipfile
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

data_folder = "data/test/20240212"

if not os.path.isdir(os.path.join(data_folder, "output")):
    os.mkdir(os.path.join(data_folder, "output"))

score_data_ori = pd.read_csv(
    os.path.join(data_folder, "score.csv"), dtype={"uuid": str}
)

uuid_list = [str(i) for i in range(14815, 17939)] + [
    "0" + str(i) for i in range(4008, 4472)
]

for wc_type in ["char", "word", "emb"]:
    # for metric in ["p" ,"r", "f", "s"]:
    for metric in ["f", "s"]:
        if wc_type == "emb" and metric != "s":
            continue
        if wc_type != "emb" and metric == "s":
            continue
        for method in ["a"]:
            # for method  in ["hqd", "d", "a"]:
            print(wc_type + "_" + metric + "_" + method)
            best_index_before = 2

            score_data = score_data_ori[
                (score_data_ori["level"] == wc_type)
                & (score_data_ori["metric"] == metric)
            ]

            # print(score_data)

            data_list = []

            for file in sorted(os.listdir(data_folder)):
                if not file.endswith(".jsonl"):
                    continue
                f = open(os.path.join(data_folder, file))
                data = f.readlines()
                result_data = []
                for i in data:
                    result_data.append(json.loads(i)["response"])
                data_list.append(result_data)
                f.close()

            for emb_num in range(len(data_list), 2, -1):

                print(emb_num)

                choose_data_list = list(
                    itertools.combinations([i for i in range(len(data_list))], emb_num)
                )
                for emb_comb in choose_data_list:
                    emb_comb = list(emb_comb)

                    sub_name = (
                        str(len(emb_comb)) + "_" + "_".join([str(i) for i in emb_comb])
                    )

                    final_data = []

                    calculate_names = list(itertools.combinations(emb_comb, 2))

                    for i in tqdm(range(len(data_list[0]))):
                        uuid = uuid_list[i]
                        sub_score_data = score_data[score_data["uuid"] == uuid]
                        best_index = best_index_before
                        best_score = -1
                        # if method == "hqd":
                        #     for j in range(len(data_list)):
                        #         score = sub_score_data.loc[sub_score_data['type']=="hqd_a_" + str(j),'score'].item()
                        #         if score > best_score:
                        #             best_score = score
                        #             best_index = j
                        # elif method == "d":
                        #     for j in range(len(data_list)):
                        #         score = sub_score_data.loc[sub_score_data['type']=="d_a_" + str(j),'score'].item()
                        #         if score > best_score:
                        #             best_score = score
                        #             best_index = j
                        # elif method == "a":
                        if method == "a":
                            matrix = np.ones(shape=(emb_num, emb_num))
                            for pair in calculate_names:
                                p0, p1 = pair[0], pair[1]
                                score = sub_score_data.loc[
                                    sub_score_data["type"]
                                    == "a_" + str(p0) + "_a_" + str(p1),
                                    "score",
                                ].item()
                                matrix[emb_comb.index(p0)][emb_comb.index(p1)] = score
                                matrix[emb_comb.index(p1)][emb_comb.index(p0)] = score
                            best_index = emb_comb[np.argmax(matrix.sum(axis=0))]
                        else:
                            print("No such method")
                            exit()
                        # print(best_index)
                        temp_json = {}
                        temp_json["uuid"] = uuid
                        temp_json["prediction"] = data_list[best_index][i]
                        final_data.append(temp_json)

                    with open("submission.json", "w") as f:
                        json.dump(final_data, f, ensure_ascii=False, indent=4)

                    zip_file = zipfile.ZipFile(
                        os.path.join(
                            data_folder,
                            "output",
                            wc_type
                            + "_"
                            + method
                            + "_"
                            + metric
                            + "_"
                            + sub_name
                            + ".zip",
                        ),
                        "w",
                    )
                    zip_file.write(
                        filename="submission.json",
                        arcname="submission.json",
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                    zip_file.close()

os.remove("submission.json")
