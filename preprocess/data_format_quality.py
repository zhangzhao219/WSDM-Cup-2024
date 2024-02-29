import json
import pandas as pd
from tqdm import tqdm

quality_score = 0.95

mode = "quality/" + str(quality_score)

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

quality_csv_data = pd.read_csv("answer_quality.csv", dtype={"uuid": str})
quality_csv_data = quality_csv_data[(quality_csv_data["quality"] >= quality_score) | (quality_csv_data["quality"] == -1)]
quality_uuid = quality_csv_data["uuid"].tolist()

final_data_train = []

for index, data in enumerate(tqdm(train_data)):
    if data["uuid"] not in quality_uuid:
        continue
    data_dict = {}
    documents_list = data["documents"]
    data_dict["query"] = data["question"] + "\n" + "\n".join(documents_list)
    data_dict["response"] = data["answer"]
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list
    final_data_train.append(data_dict)

print(len(final_data_train))

with open("../data/wsdm/model/" + mode + "/release_train_data.json", "w") as f:
    json.dump(final_data_train, f, ensure_ascii=False, indent=4)

