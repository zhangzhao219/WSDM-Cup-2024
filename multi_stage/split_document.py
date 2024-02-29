import json
from tqdm import tqdm

split_num = 10

for split_order in range(split_num):
    with open(
        "../data/wsdm/prepare/split/"
        + str(split_num)
        + "/"
        + str(split_num)
        + "_eval_"
        + str(split_order)
        + ".json",
        "r",
    ) as f:
        eval_data = json.load(f)

    final_data = []

    num_order_dict = {}

    for data in tqdm(eval_data):
        num_order_dict[data["uuid"]] = []
        data_dict = {}
        data_dict["response"] = ""
        history_list = []
        for i in data["history"]:
            history_list.append([i["question"], i["answer"]])
        data_dict["history"] = history_list

        now_have_data = False
        for i, document in enumerate(data["documents"]):
            if len(document) == 0:
                continue
            now_have_data = True
            num_order_dict[data["uuid"]].append(i)
            data_dict["query"] = data["question"] + "\n" + document + "\n"
            final_data.append(data_dict.copy())
        if not now_have_data:
            num_order_dict[data["uuid"]].append(-1)
            data_dict["query"] = data["question"] + "\n"
            final_data.append(data_dict.copy())

    print(len(final_data))

    with open(
        "../data/wsdm/model/split_document/"
        + str(split_num)
        + "/"
        + str(split_num)
        + "_eval_"
        + str(split_order)
        + ".json",
        "w",
    ) as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    with open(
        "../data/wsdm/model/split_document/"
        + str(split_num)
        + "/"
        + str(split_num)
        + "_eval_num_order_"
        + str(split_order)
        + ".json",
        "w",
    ) as f:
        json.dump(num_order_dict, f, ensure_ascii=False, indent=4)



with open("../data/wsdm/prepare/split/"+ str(split_num)+"/release_phase1_eval_data_wo_gt.json") as f:
    real_eval_data = json.load(f)

final_data = []

num_order_dict = {}

for data in tqdm(real_eval_data):
    num_order_dict[data["uuid"]] = []
    data_dict = {}
    data_dict["response"] = ""
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list

    now_have_data = False
    for i, document in enumerate(data["documents"]):
        if len(document) == 0:
            continue
        now_have_data = True
        num_order_dict[data["uuid"]].append(i)
        data_dict["query"] = data["question"] + "\n" + document + "\n"
        final_data.append(data_dict.copy())
    if not now_have_data:
        num_order_dict[data["uuid"]].append(-1)
        data_dict["query"] = data["question"] + "\n"
        final_data.append(data_dict.copy())
            
print(len(final_data))

with open("../data/wsdm/model/split_document/"+ str(split_num)+"/release_phase1_eval_data_wo_gt.json","w",) as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

with open(
    "../data/wsdm/model/split_document/"
    + str(split_num)
    + "/release_phase1_eval_data_wo_gt_num_order.json",
    "w",
) as f:
    json.dump(num_order_dict, f, ensure_ascii=False, indent=4)