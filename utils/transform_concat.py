import json

with open("../data/wsdm/prepare/ori/release_phase2_test_data_wo_gt.json", "r") as f:
    test_data = json.load(f)

uuid_list = [str(i) for i in range(14815,17939)] + ["0" + str(i) for i in range(4008,4472)]

final_data = []

for i, data in enumerate(test_data):
    temp_json = {}
    temp_json["uuid"] = uuid_list[i]
    history_question_answer = ""
    for i, h in enumerate(data["history"]):
        history_question_answer = history_question_answer + h["question"] + "\n"
        history_question_answer = history_question_answer + h["answer"] + "\n"
    temp_json["prediction"] = history_question_answer + "\n" + data["question"] + "\n" + "\n".join(data["documents"])
    final_data.append(temp_json)

with open("submission_clean.json", "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
