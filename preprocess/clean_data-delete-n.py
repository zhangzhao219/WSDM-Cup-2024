import re
import json
from tqdm import tqdm


def deal_text(text):
    # 空格处理
    no_n = text.replace("\n", " ")
    no_n = re.sub(" +", " ", no_n)
    return no_n


with open("../data/wsdm/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

for data in tqdm(train_data):
    document_list = []
    for i, document in enumerate(data["documents"]):
        if len(document) == 0:
            continue
        document_list.append(deal_text(document))

    data["documents"] = document_list


with open("../data/wsdm/delete-n/release_train_data.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)


with open("../data/wsdm/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

for data in tqdm(eval_data):
    document_list = []
    for i, document in enumerate(data["documents"]):
        if len(document) == 0:
            continue
        document_list.append(deal_text(document))

    data["documents"] = document_list

with open("../data/wsdm/delete-n/release_phase1_eval_data_wo_gt.json", "w") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)
