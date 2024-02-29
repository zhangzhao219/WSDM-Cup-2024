import json
from tqdm import tqdm

mode = "Pseudo/phase_1/shmily/infer_result_20240206-182534"

phase_2_switch = False

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

with open("../data/wsdm/other/shmily/infer_result_20240206-182534.jsonl", "r") as f:
    best_eval_answer = f.readlines()

if phase_2_switch:
    with open("../data/wsdm/other/shmily/infer_result_20240205-183317.jsonl", "r") as f:
        best_test_answer = f.readlines()

final_data_train = []

for index, data in enumerate(tqdm(train_data)):
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

with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

final_data_eval = []

for index, data in enumerate(tqdm(eval_data)):
    data_dict = {}
    documents_list = data["documents"]
    data_dict["query"] = data["question"] + "\n" + "\n".join(documents_list)
    data_dict["response"] = json.loads(best_eval_answer[index])["response"]
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list
    final_data_eval.append(data_dict)

print(len(final_data_eval))

final_data_train.extend(final_data_eval)

with open("../data/wsdm/prepare/ori/release_phase2_test_data_wo_gt.json", "r") as f:
    test_data = json.load(f)

if phase_2_switch:
    final_data_test = []

    for index, data in enumerate(tqdm(test_data)):
        data_dict = {}
        documents_list = data["documents"]
        data_dict["query"] = data["question"] + "\n" + "\n".join(documents_list)
        data_dict["response"] = json.loads(best_test_answer[index])["response"]
        history_list = []
        for i in data["history"]:
            history_list.append([i["question"], i["answer"]])
        data_dict["history"] = history_list
        final_data_test.append(data_dict)

    print(len(final_data_test))

    final_data_train.extend(final_data_test)

print(len(final_data_train))

with open("../data/wsdm/model/" + mode + "/release_train_data.json", "w") as f:
    json.dump(final_data_train, f, ensure_ascii=False, indent=4)
