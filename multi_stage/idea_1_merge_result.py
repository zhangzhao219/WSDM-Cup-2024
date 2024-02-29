import os
import json
from tqdm import tqdm

eval_data_name = "release_phase1_eval_data_wo_gt.json"
train_data_name = "release_train_data.json"

split_num = 10

ori_file_path = "../data/wsdm/prepare/split/" + str(split_num)
idea_1_file_path = "idea_1/20240118-1000"
history_switch = True
order_list = ["q", "d", "p"]

folder_name = "".join(order_list)
if history_switch:
    folder_name += "_h"

with open(os.path.join(ori_file_path, eval_data_name), "r") as f:
    eval_data = json.load(f)

final_train_data = []

for infer_file in os.listdir(idea_1_file_path):
    if not infer_file.startswith("infer"):
        continue
    now_file_order = infer_file.split("_")[-1].split(".")[0]
    if now_file_order == "-1":
        with open(os.path.join(idea_1_file_path, infer_file), "r") as f:
            now_data = f.readlines()
        for i in tqdm(range(len(eval_data))):
            eval_data[i]["predict"] = json.loads(now_data[i])["response"]
        continue
    with open(
        os.path.join(
            ori_file_path, str(split_num) + "_eval_" + now_file_order + ".json"
        ),
        "r",
    ) as f:
        ori_data = json.load(f)
    with open(os.path.join(idea_1_file_path, infer_file), "r") as f:
        now_data = f.readlines()

    for i in tqdm(range(len(ori_data))):
        d = ori_data[i]
        d["predict"] = json.loads(now_data[i])["response"]
        final_train_data.append(d)

final_train_data = sorted(final_train_data, key=lambda x: x["uuid"], reverse=False)

with open(os.path.join(idea_1_file_path, train_data_name), "w") as f:
    json.dump(final_train_data, f, ensure_ascii=False, indent=4)

with open(os.path.join(idea_1_file_path, eval_data_name), "w") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)


final_save_train_data = []

for data in tqdm(final_train_data):
    data_dict = {}

    data_dict["response"] = data["answer"]
    data_dict["history"] = []

    if history_switch:
        history_list = []
        for i in data["history"]:
            history_list.append([i["question"], i["answer"]])
        data_dict["history"] = history_list

    data_dict["query"] = ""

    for mode in order_list:
        if mode == "q":
            data_dict["query"] = data_dict["query"] + data["question"] + "\n"
        elif mode == "d":
            data_dict["query"] = (
                data_dict["query"] + "\n".join(data["documents"]) + "\n"
            )
        elif mode == "p":
            data_dict["query"] = data_dict["query"] + data["predict"] + "\n"
        else:
            print("Unknown mode")
            exit()

    final_save_train_data.append(data_dict)

final_save_eval_data = []

for data in tqdm(eval_data):
    data_dict = {}

    data_dict["response"] = ""
    data_dict["history"] = []

    if history_switch:
        history_list = []
        for i in data["history"]:
            history_list.append([i["question"], i["answer"]])
        data_dict["history"] = history_list

    data_dict["query"] = ""

    for mode in order_list:
        if mode == "q":
            data_dict["query"] = data_dict["query"] + data["question"] + "\n"
        elif mode == "d":
            data_dict["query"] = (
                data_dict["query"] + "\n".join(data["documents"]) + "\n"
            )
        elif mode == "p":
            data_dict["query"] = data_dict["query"] + data["predict"] + "\n"
        else:
            print("Unknown mode")
            exit()

    final_save_eval_data.append(data_dict)

save_data_folder = (
    "../data/wsdm/model/multi_stage/idea_1/"
    + str(split_num)
    + "/"
    + idea_1_file_path.split("/")[-1]
    + "/"
    + folder_name
)

if not os.path.isdir(save_data_folder):
    os.makedirs(save_data_folder)

with open(os.path.join(save_data_folder, train_data_name), "w") as f:
    json.dump(final_save_train_data, f, ensure_ascii=False, indent=4)

with open(os.path.join(save_data_folder, eval_data_name), "w") as f:
    json.dump(final_save_eval_data, f, ensure_ascii=False, indent=4)
