import json
from tqdm import tqdm

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

final_data = []

for data in tqdm(train_data):
    final_data.append(data)

# for data in tqdm(eval_data):
#     document_list = []
#     history_len = len(data["history"])
#     if history_len == 0:
#         continue
#     history_list = []
#     for i, his in enumerate(data["history"]):
#         if i == history_len-1:
#             data["question"] = his["question"]
#             data["answer"] = his["answer"]
#         else:
#             history_list.append(his)
#     data["history"] = history_list
#     data["documents"] = []
#     final_data.append(data)

for data in tqdm(eval_data):
    history_len = len(data["history"])
    if history_len == 0:
        continue
    data["documents"] = []
    data["question"] = ""
    final_data.append(data)

with open("../data/wsdm/prepare/train_eval/release_train_eval_data_only_history.json", "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
