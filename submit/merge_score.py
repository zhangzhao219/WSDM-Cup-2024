import os
import json
import zipfile
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

data_folder = "merge"

score_data_ori = pd.read_csv(
    os.path.join(data_folder, "score.csv"), dtype={"uuid": str}
)

uuid_list = [str(i) for i in range(14815, 17939)] + [
    "0" + str(i) for i in range(4008, 4472)
]

for wc_type in ["emb"]:
    for metric in ["s"]:
        for method in ["a"]:
            print(wc_type + "_" + metric + "_" + method)
            best_index_before = 2

            score_data = score_data_ori[
                (score_data_ori["level"] == wc_type)
                & (score_data_ori["metric"] == metric)
            ]

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

            for emb_num in [len(data_list)]:

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
                        wc_type + "_" + method + "_" + metric + "_" + sub_name + ".zip",
                        "w",
                    )
                    zip_file.write(
                        filename="submission.json",
                        arcname="submission.json",
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                    zip_file.close()

os.remove("submission.json")
