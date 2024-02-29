import os
import json
import pandas as pd
from tqdm import tqdm

metrics_dict = {
    "embedding_q": {"max": 0.9, "min": 0.2},
    "embedding_hqaq": {"max": 0.95, "min": 0.25},
    "rouge_word_q_f": {"max": 0.3, "min": -1},
    "rouge_word_hqaq_f": {"max": 0.3, "min": -1},
    "rouge_character_q_f": {"max": 0.5, "min": -1},
    "rouge_character_hqaq_f": {"max": 0.5, "min": -1},
}

show_example = True
filter_mode = "ANY"
# "ALL","ANY","VOTE"

# embedding_q,embedding_hqq,embedding_hqaq,embedding_a

# rouge_word_q_p,rouge_word_q_r,rouge_word_q_f
# rouge_word_hqq_p,rouge_word_hqq_r,rouge_word_hqq_f
# rouge_word_hqaq_p,rouge_word_hqaq_r,rouge_word_hqaq_f
# rouge_word_a_p,rouge_word_a_r,rouge_word_a_f

# rouge_character_q_p,rouge_character_q_r,rouge_character_q_f
# rouge_character_hqq_p,rouge_character_hqq_r,rouge_character_hqq_f
# rouge_character_hqaq_p,rouge_character_hqaq_r,rouge_character_hqaq_f
# rouge_character_a_p,rouge_character_a_r,rouge_character_a_f

# bm25_q,bm25_hqq,bm25_hqaq,bm25_a

name = ""
if name == "":
    for k, v in metrics_dict.items():
        name += k + "_" + str(v["max"]) + "_" + str(v["min"]) + "-"
    name = name + filter_mode

dest_folder = "delete-doc/" + name
if not os.path.exists("../data/wsdm/prepare/" + dest_folder):
    os.makedirs("../data/wsdm/prepare/" + dest_folder)
else:
    print("Folder Exists!")


with open("../data/wsdm/prepare/ori/release_phase2_test_data_wo_gt.json", "r") as f:
    test_data = json.load(f)


score_train_test_csv = pd.read_csv(
    "../data/wsdm/score_train_test.csv", dtype={"uuid": str}
).fillna(-1)

info_column = ["uuid", "data", "type"]

metrics_keys_list = list(metrics_dict.keys())
score_csv = score_train_test_csv[info_column + metrics_keys_list]

score_csv_temp = score_csv[score_csv["data"] == "test"]
for key in metrics_keys_list:
    print("Max Score:", score_csv_temp[key].max())
    print("Min Score:", score_csv_temp[score_csv_temp[key] != -1][key].min())
    print(score_csv_temp[score_csv_temp[key] != -1][key].value_counts(bins=10))

total_delete = 0

for index, data in enumerate(tqdm(test_data)):
    uuid = data["uuid"]
    documents_list = data["documents"]

    score_uuid = score_csv[score_csv["uuid"] == uuid][metrics_keys_list]

    score_dict = {}

    for metric_key, metric_value in metrics_dict.items():
        metric_value_max = metric_value["max"]
        metric_value_min = metric_value["min"]
        score_dict[metric_key] = score_uuid[metric_key].tolist()
        score_uuid[metric_key] = (score_uuid[metric_key] <= metric_value_max) & (
            score_uuid[metric_key] >= metric_value_min
        )
    # print(score_uuid)

    if len(documents_list) != len(score_uuid):
        print(uuid)
        print(len(documents_list), len(score_uuid))
        exit()

    score_key_judge = score_uuid.sum(axis=1).tolist()
    # print(score_key_judge)
    # print(score_dict)

    score_judge = []
    if filter_mode == "ANY":
        for i in score_key_judge:
            if i != len(metrics_keys_list):
                score_judge.append(False)
            else:
                score_judge.append(True)
    elif filter_mode == "ALL":
        for i in score_key_judge:
            if i == 0:
                score_judge.append(False)
            else:
                score_judge.append(True)
    elif filter_mode == "VOTE":
        for i in score_key_judge:
            if i < len(metrics_keys_list) / 2:
                score_judge.append(False)
            else:
                score_judge.append(True)
    else:
        print("Unknown mode")
        exit()
    # print(score_judge)
    # exit()

    final_document_list = []
    for i, d in enumerate(documents_list):
        if score_judge[i]:
            final_document_list.append(d)
        else:
            if d == "":
                continue
            total_delete += 1
            if show_example:
                print("-------------------------------------------------")
                for iqa, qa in enumerate(data["history"]):
                    print("History Question", iqa, ":", qa["question"])
                    print("History Answer", iqa, ":", qa["answer"])
                print("Question:", data["question"])
                print("-------------------------------------------------")
                print("Document:", d)
                for k in metrics_keys_list:
                    print(k, "Score:", score_dict[k][i])
                print("-------------------------------------------------")
                if input("Delete? Y/N (Default Y): ") == "N":
                    final_document_list.append(d)

    test_data[index]["documents"] = final_document_list

print("Delete", total_delete, "Documents")
with open(
    "../data/wsdm/prepare/" + dest_folder + "/release_phase2_test_data_wo_gt.json",
    "w",
) as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)


# # train and eval data and Deprecated Reorder

# reorder = False

# train_data_switch = False
# eval_data_switch = False

# with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
#     train_data = json.load(f)

# with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
#     eval_data = json.load(f)

# score_train_eval_csv = pd.read_csv(
#     "../data/wsdm/score_train_eval.csv", dtype={"uuid": str}
# ).fillna(-1)

# if train_data_switch:
#     score_csv_temp = score_csv[score_csv["data"] == "train"]
#     print(score_csv_temp[metrics[0]].value_counts(bins=10))
#     print("Max Score:", score_csv_temp[metrics[0]].max())
#     print("Min Score:", score_csv_temp[metrics[0]].min())

#     for index, data in enumerate(tqdm(train_data)):
#         uuid = data["uuid"]
#         score_i = score_csv[score_csv["uuid"] == uuid]
#         documents_list = data["documents"]

#         for metric in metrics:
#             score_this_metric = score_i[metric].tolist()

#             if len(documents_list) != len(score_this_metric):
#                 print(uuid)
#                 print(len(documents_list), len(score_this_metric))
#                 exit()

#             metric_to_document = zip(
#                 score_this_metric, [i for i in range(0, len(score_this_metric))]
#             )
#             metric_to_document = [(m, i) for (m, i) in metric_to_document]

#             if refilter:
#                 metric_to_document_update = []
#                 for m, i in metric_to_document:
#                     if m <= max_metric and m >= min_metric:
#                         metric_to_document_update.append((m, i))
#                 metric_to_document = metric_to_document_update
#             if reorder:
#                 metric_to_document = sorted(
#                     metric_to_document, key=lambda x: x[0], reverse=True
#                 )

#             train_data[index]["documents"] = [
#                 documents_list[k[1]] for k in metric_to_document
#             ]

#     with open(
#         "../data/wsdm/prepare/" + dest_folder + "/release_train_data.json", "w"
#     ) as f:
#         json.dump(train_data, f, ensure_ascii=False, indent=4)

# if eval_data_switch:
#     score_csv_temp = score_csv[score_csv["data"] == "eval"]
#     print(score_csv_temp[metrics[0]].value_counts(bins=10))
#     print("Max Score:", score_csv_temp[metrics[0]].max())
#     print("Min Score:", score_csv_temp[metrics[0]].min())

#     for index, data in enumerate(tqdm(eval_data)):
#         uuid = data["uuid"]
#         score_i = score_csv[score_csv["uuid"] == uuid]
#         documents_list = data["documents"]

#         for metric in metrics:
#             score_this_metric = score_i[metric].tolist()

#             if len(documents_list) != len(score_this_metric):
#                 print(uuid)
#                 print(len(documents_list), len(score_this_metric))
#                 exit()

#             metric_to_document = zip(
#                 score_this_metric, [i for i in range(0, len(score_this_metric))]
#             )
#             metric_to_document = [(m, i) for (m, i) in metric_to_document]

#             # print(metric_to_document)

#             if refilter:
#                 metric_to_document_update = []
#                 for m, i in metric_to_document:
#                     if m <= max_metric and m >= min_metric:
#                         metric_to_document_update.append((m, i))
#                     else:
#                         if show_example:
#                             if m == 0 or m == -1:
#                                 continue
#                             print("-------------------------------------------------")
#                             for iqa, qa in enumerate(data["history"]):
#                                 print("History Question", iqa, ":", qa["question"])
#                                 print("History Answer", iqa, ":", qa["answer"])
#                             print("Question:", data["question"])
#                             print("-------------------------------------------------")
#                             print("Document:", documents_list[i])
#                             print("Score:", m)
#                             print("-------------------------------------------------")
#                             xy = input()
#                 metric_to_document = metric_to_document_update
#                 # print(metric_to_document)
#             if reorder:
#                 metric_to_document = sorted(
#                     metric_to_document, key=lambda x: x[0], reverse=True
#                 )
#                 # print(metric_to_document)

#             eval_data[index]["documents"] = [
#                 documents_list[k[1]] for k in metric_to_document
#             ]

#     with open(
#         "../data/wsdm/prepare/" + dest_folder + "/release_phase1_eval_data_wo_gt.json",
#         "w",
#     ) as f:
#         json.dump(eval_data, f, ensure_ascii=False, indent=4)
