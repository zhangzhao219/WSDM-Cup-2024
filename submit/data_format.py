import json
from tqdm import tqdm

with open("data/release_phase2_test_data_wo_gt.json", "r") as f:
    test_data = json.load(f)

final_data = []

for index, data in enumerate(tqdm(test_data)):
    data_dict = {}
    documents_list = data["documents"]
    data_dict["query"] = data["question"] + "\n" + "\n".join(documents_list)
    data_dict["response"] = ""
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list
    final_data.append(data_dict)

with open("release_phase2_test_data_wo_gt.json", "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
