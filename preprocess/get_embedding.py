import re
import json
from tqdm import tqdm
from nltk import sent_tokenize

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

print(len(train_data))

with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

print(len(eval_data))

body_list = []
title_list = []


def combine_json(metaid, sentence):
    now_json = {}
    now_json["metadata"] = metaid
    now_json["market"] = "en-us"
    now_json["context"] = sentence
    return now_json


for data in tqdm(train_data):
    uuid = data["uuid"]

    # for i, d in enumerate(data["documents"]):
    #     body_list.append(combine_json(uuid + "-train-documents-" + str(i), d))
    #     title_list.append(combine_json(uuid + "-train-documents-" + str(i), "test"))

    # body_list.append(combine_json(uuid + "-train-question", data["question"]))
    # title_list.append(combine_json(uuid + "-train-question", "test"))

    # body_list.append(combine_json(uuid + "-train-answer", data["answer"]))
    # title_list.append(combine_json(uuid + "-train-answer", "test"))

    # for i, h in enumerate(data["history"]):
    #     body_list.append(
    #         combine_json(uuid + "-train-history-question-" + str(i), h["question"])
    #     )
    #     body_list.append(
    #         combine_json(uuid + "-train-history-answer-" + str(i), h["answer"])
    #     )

    #     title_list.append(
    #         combine_json(uuid + "-train-history-question-" + str(i), "test")
    #     )
    #     title_list.append(
    #         combine_json(uuid + "-train-history-answer-" + str(i), "test")
    #     )

    # history_question_answer = ""
    # for i, h in enumerate(data["history"]):
    #     history_question_answer = history_question_answer + h["question"] + "\n"
    #     history_question_answer = history_question_answer + h["answer"] + "\n"

    # body_list.append(
    #     combine_json(uuid + "-train-history-question-and-answer", history_question_answer)
    # )

    # title_list.append(
    #     combine_json(uuid + "-train-history-question-and-answer", "test")
    # )

    # all_question_answer = ""
    # all_question = ""
    # for i, h in enumerate(data["history"]):
    #     all_question_answer = all_question_answer + h["question"] + "\n"
    #     all_question = all_question + h["question"] + "\n"
    #     all_question_answer = all_question_answer + h["answer"] + "\n"
    
    # all_question = all_question + data["question"]
    # all_question_answer = all_question_answer + data["question"]

    # body_list.append(
    #     combine_json(uuid + "-train-all-question-and-answer", all_question_answer)
    # )

    # body_list.append(
    #     combine_json(uuid + "-train-all-question", all_question)
    # )

    # title_list.append(
    #     combine_json(uuid + "-train-all-question-and-answer", "test")
    # )

    # title_list.append(
    #     combine_json(uuid + "-train-all-question", "test")
    # )

    for i, d in enumerate(data["documents"]):
        # sentence_list = re.split("\n\t\n|\n\n",d)
        sentence_list = sent_tokenize(d)
        for j,sentence in enumerate(sentence_list):
            if len(sentence) <= 10:
                continue
            body_list.append(combine_json(uuid + "-train-documents-" + str(i) + "-split-" + str(j), sentence))
            title_list.append(combine_json(uuid + "-train-documents-" + str(i) + "-split-" + str(j), "test"))

for data in tqdm(eval_data):
    uuid = data["uuid"]

    # for i, d in enumerate(data["documents"]):
    #     body_list.append(combine_json(uuid + "-eval-documents-" + str(i), d))
    #     title_list.append(combine_json(uuid + "-eval-documents-" + str(i), "test"))

    # body_list.append(combine_json(uuid + "-eval-question", data["question"]))
    # title_list.append(combine_json(uuid + "-eval-question", "test"))


    # for i, h in enumerate(data["history"]):
    #     body_list.append(
    #         combine_json(uuid + "-eval-history-question-" + str(i), h["question"])
    #     )
    #     body_list.append(
    #         combine_json(uuid + "-eval-history-answer-" + str(i), h["answer"])
    #     )

    #     title_list.append(
    #         combine_json(uuid + "-eval-history-question-" + str(i), "test")
    #     )
    #     title_list.append(
    #         combine_json(uuid + "-eval-history-answer-" + str(i), "test")
    #     )


    # history_question_answer = ""
    # for i, h in enumerate(data["history"]):
    #     history_question_answer = history_question_answer + h["question"] + "\n"
    #     history_question_answer = history_question_answer + h["answer"] + "\n"

    # body_list.append(
    #     combine_json(uuid + "-eval-history-question-and-answer", history_question_answer)
    # )

    # title_list.append(
    #     combine_json(uuid + "-eval-history-question-and-answer", "test")
    # )

    # all_question_answer = ""
    # all_question = ""
    # for i, h in enumerate(data["history"]):
    #     all_question_answer = all_question_answer + h["question"] + "\n"
    #     all_question = all_question + h["question"] + "\n"
    #     all_question_answer = all_question_answer + h["answer"] + "\n"
    
    # all_question = all_question + data["question"]
    # all_question_answer = all_question_answer + data["question"]

    # body_list.append(
    #     combine_json(uuid + "-eval-all-question-and-answer", all_question_answer)
    # )

    # body_list.append(
    #     combine_json(uuid + "-eval-all-question", all_question)
    # )

    # title_list.append(
    #     combine_json(uuid + "-eval-all-question-and-answer", "test")
    # )

    # title_list.append(
    #     combine_json(uuid + "-eval-all-question", "test")
    # )

    for i, d in enumerate(data["documents"]):
        # sentence_list = re.split("\n\t\n|\n\n",d)
        sentence_list = sent_tokenize(d)
        for j,sentence in enumerate(sentence_list):
            if len(sentence) <= 10:
                continue
            body_list.append(combine_json(uuid + "-eval-documents-" + str(i) + "-split-" + str(j), sentence))
            title_list.append(combine_json(uuid + "-eval-documents-" + str(i) + "-split-" + str(j), "test"))

with open("title.jsonl", "w") as f:
    for i in title_list:
        f.write(json.dumps(i, ensure_ascii=False) + "\n")

with open("body.jsonl", "w") as f:
    for i in body_list:
        f.write(json.dumps(i, ensure_ascii=False) + "\n")
