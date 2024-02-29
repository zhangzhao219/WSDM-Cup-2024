import json
from tqdm import tqdm

mode = "ori"

# with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
#     train_data = json.load(f)

# with open("../keyword/3_5_keyword_train.json", "r") as f:
#     gpt_3_5_keyword_train = json.load(f)

# with open("../keyword/4_keyword_train.json", "r") as f:
#     gpt_4_keyword_train = json.load(f)

# final_data = []

# for index, data in enumerate(tqdm(train_data)):
#     data_dict = {}
#     documents_list = data["documents"]
#     if mode == "question-tail":
#         data_dict["query"] = (
#             data["question"]
#             + "\n"
#             + "\n".join(documents_list)
#             + "\n"
#             + data["question"]
#         )
#     elif mode == "gpt-3-5-keyword-tail":
#         data_dict["query"] = (
#             data["question"]
#             + "\n"
#             + "\n".join(documents_list)
#             + "\n"
#             + gpt_3_5_keyword_train[index]["prediction"]
#         )
#     elif mode == "gpt-4-keyword-tail":
#         data_dict["query"] = (
#             data["question"]
#             + "\n"
#             + "\n".join(documents_list)
#             + "\nKeyword:"
#             + gpt_4_keyword_train[index]["prediction"].replace('"', "")
#         )
#     else:
#         data_dict["query"] = data["question"] + "\n" + "\n".join(documents_list)
#     data_dict["response"] = data["answer"]
#     history_list = []
#     for i in data["history"]:
#         history_list.append([i["question"], i["answer"]])
#     data_dict["history"] = history_list
#     final_data.append(data_dict)

# print(len(final_data))

# # with open("../data/wsdm/model/ori/release_train_data.json", "w") as f:
# #     json.dump(final_data, f, ensure_ascii=False, indent=4)

# with open("../data/wsdm/model/plus/" + mode + "/release_train_data.json", "w") as f:
#     json.dump(final_data, f, ensure_ascii=False, indent=4)


with open("../data/wsdm/prepare/delete-doc/embedding_q_0.9_0.2-embedding_hqaq_0.95_0.25-rouge_word_q_f_0.3_-1-rouge_word_hqaq_f_0.3_-1-rouge_character_q_f_0.5_-1-rouge_character_hqaq_f_0.5_-1-ANY/release_phase2_test_data_wo_gt.json", "r") as f:
    test_data = json.load(f)

# with open("../keyword/3_5_keyword_eval.json", "r") as f:
#     gpt_3_5_keyword_eval = json.load(f)

# with open("../keyword/4_keyword_eval.json", "r") as f:
#     gpt_4_keyword_eval = json.load(f)

final_data = []

for index, data in enumerate(tqdm(test_data)):
    # print(data["uuid"])
    data_dict = {}
    documents_list = data["documents"]
    if mode == "question-tail":
        data_dict["query"] = (
            data["question"]
            + "\n"
            + "\n".join(documents_list)
            + "\n"
            + data["question"]
        )
    elif mode == "gpt-3-5-keyword-tail":
        data_dict["query"] = (
            data["question"]
            + "\n"
            + "\n".join(documents_list)
            + "\n"
            + gpt_3_5_keyword_eval[index]["prediction"]
        )
    elif mode == "gpt-4-keyword-tail":
        data_dict["query"] = (
            data["question"]
            + "\n"
            + "\n".join(documents_list)
            + "\nKeyword:"
            + gpt_4_keyword_eval[index]["prediction"].replace('"', "")
        )
    else:
        data_dict["query"] = data["question"] + "\n" + "\n".join(documents_list)
    data_dict["response"] = ""
    history_list = []
    for i in data["history"]:
        history_list.append([i["question"], i["answer"]])
    data_dict["history"] = history_list
    final_data.append(data_dict)

print(len(final_data))

# with open("../data/wsdm/model/ori/release_phase1_eval_data_wo_gt.json", "w") as f:
#     json.dump(final_data, f, ensure_ascii=False, indent=4)

# with open(
#     "../data/wsdm/model/plus/" + mode + "/release_phase2_test_data_wo_gt.json", "w"
# ) as f:
#     json.dump(final_data, f, ensure_ascii=False, indent=4)

with open(
    "../data/wsdm/model/delete-doc/release_phase2_test_data_wo_gt.json", "w"
) as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)