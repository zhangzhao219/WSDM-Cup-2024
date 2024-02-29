import json

with open("../data/wsdm/model/multi_stage/idea_1/10/20240118-1000/qdp_h/release_train_data.json","r") as f:
    data_summary = json.load(f)

with open("../data/wsdm/model/ori/release_train_data.json","r") as f:
    data_ori = json.load(f)

final_data = []

for data in data_summary:
    final_data.append(data)

for data in data_ori:
    final_data.append(data)

with open(
    "../data/wsdm/model/multi_stage/mix/release_train_data.json", "w"
) as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
