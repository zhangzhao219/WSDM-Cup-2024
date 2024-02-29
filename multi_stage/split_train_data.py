import random
import json
from tqdm import tqdm

num_split = 10

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)


number_list = [i for i in range(len(train_data))]
random.seed(42)
random.shuffle(number_list)  # 打乱列表顺序
number_bins = [number_list[i::num_split] for i in range(num_split)]

data_train_list = [[] for i in range(num_split)]
data_val_list = [[] for i in range(num_split)]


for i, data in enumerate(tqdm(train_data)):
    for j, number_bin in enumerate(number_bins):
        if i not in number_bin:
            data_train_list[j].append(data)
        else:
            data_val_list[j].append(data)

for i, data_train in enumerate(data_train_list):
    with open(
        "../data/wsdm/prepare/split/" + str(num_split) + "/" + str(num_split) + "_train_" + str(i) + ".json",
        "w",
    ) as f:
        json.dump(data_train, f, ensure_ascii=False, indent=4)

for i, data_val in enumerate(data_val_list):
    with open(
        "../data/wsdm/prepare/split/" + str(num_split) + "/" + str(num_split) + "_eval_" + str(i) + ".json",
        "w",
    ) as f:
        json.dump(data_val, f, ensure_ascii=False, indent=4)

with open(
    "../data/wsdm/prepare/split/" + str(num_split) + "/release_phase1_eval_data_wo_gt.json",
    "w",
) as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=4)