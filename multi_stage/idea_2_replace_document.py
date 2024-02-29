import os
import sys
import json
from tqdm import tqdm

eval_data_name = "release_phase1_eval_data_wo_gt.json"
train_data_name = "release_train_data.json"

idea_2_file_path = "idea_2/20240118-1000"

split_num = 10

split_file_folder = "../data/wsdm/prepare/split/" + str(split_num)
dict_file_folder = "../data/wsdm/model/split_document/" + str(split_num)

final_train_data = []
final_eval_data = []

document_list_dict = {}
for file in os.listdir(dict_file_folder):
    if "order" not in file:
        continue
    with open(os.path.join(dict_file_folder, file), "r") as f:
        document_list_dict.update(json.load(f))

with open(os.path.join(split_file_folder, eval_data_name), "r") as f:
    eval_data = json.load(f)

final_train_data = []

for infer_file in os.listdir(idea_2_file_path):
    if not infer_file.startswith("infer"):
        continue
    now_file_order = infer_file.split("_")[-1].split(".")[0]
    if now_file_order == "-1":
        with open(os.path.join(idea_2_file_path, infer_file), "r") as f:
            now_data = f.readlines()
        eval_sum_len = 0
        for i in tqdm(range(len(eval_data))):
            uuid = eval_data[i]["uuid"]
            this_eval_len = len(document_list_dict[uuid])
            eval_data[i]["documents"] = [json.loads(now_data[j])["response"] for j in range(eval_sum_len, eval_sum_len+this_eval_len)]
            eval_sum_len += this_eval_len
        continue

    with open(os.path.join(split_file_folder, str(split_num) + "_eval_" + now_file_order + ".json"), "r") as f:
        ori_data = json.load(f)
    with open(os.path.join(idea_2_file_path,infer_file), "r") as f:
        now_data = f.readlines()
        
    train_sum_len = 0
    for i in tqdm(range(len(ori_data))):
        d = ori_data[i]
        uuid = d["uuid"]
        this_train_len = len(document_list_dict[uuid])
        d["documents"] = [json.loads(now_data[j])["response"] for j in range(train_sum_len, train_sum_len+this_train_len)]
        train_sum_len += this_train_len
        final_train_data.append(d)

final_train_data = sorted(final_train_data, key=lambda x: x["uuid"], reverse=False)

with open(os.path.join(idea_2_file_path,train_data_name), "w") as f:
    json.dump(final_train_data, f, ensure_ascii=False, indent=4)

with open(os.path.join(idea_2_file_path,eval_data_name), "w") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)


final_save_train_data = []

for data in tqdm(final_train_data):
    data_dict = {}
    data_dict["response"] = data["answer"]
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list
    data_dict["query"] = data["question"] + "\n" + "\n".join(data["documents"])
    final_save_train_data.append(data_dict)

final_save_eval_data = []

for data in tqdm(eval_data):
    data_dict = {}
    data_dict["response"] = ""
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list
    data_dict["query"] = data["question"] + "\n" + "\n".join(data["documents"])
    final_save_eval_data.append(data_dict)

save_data_folder = "../data/wsdm/model/multi_stage/idea_2/" + str(split_num) + "/" + idea_2_file_path.split("/")[-1]

if not os.path.isdir(save_data_folder):
	os.makedirs(save_data_folder)

with open(os.path.join(save_data_folder, train_data_name), "w") as f:
    json.dump(final_save_train_data, f, ensure_ascii=False, indent=4)

with open(os.path.join(save_data_folder, eval_data_name), "w") as f:
    json.dump(final_save_eval_data, f, ensure_ascii=False, indent=4)
