import re
import time
import json
import openai
from tqdm import tqdm
import pandas as pd

def extract_numbers_from_string(string):
    numbers = re.findall(r"\d+\.\d+|\d+", string)
    return numbers


context = "You are a helpful assistant."

context_user = "Now please read the [Answer] below, determine its quality for a system whose goal is to give users more accurate and informative instruction based on the [Information] provided, dialog history and the last [Question]. Please give a direct float value between 0 and 1 to indicate quality, with 1 indicating good quality and 0 indicating bad quality.\n"


def get_answer(context, history_list, document_list, question, answer):
    history_json_list = []
    for history in history_list:
        history_json_list.append({"role": "user", "content": history["question"]})
        history_json_list.append({"role": "assistant", "content": history["answer"]})

    context_str = " "
    for doc in document_list:
        context_str += f"[Information] {doc}\n [Question] {question}\n"

    messages = (
        [{"role": "system", "content": context}]
        + history_json_list
        + [
            {
                "role": "user",
                "content": context_str,
            },
            {
                "role": "user",
                "content": context_user + "[Answer] " + answer,
            },
        ]
    )
    # print(messages)
    response = gpt4(messages)
    res = response["choices"][0]["message"]["content"]
    # print(res)
    # exit()
    return res


if __name__ == "__main__":
    with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
        train_data = json.load(f)

    result_list = []

    for data in tqdm(train_data):
        uuid = data["uuid"]
        try_times = 10
        answer_final = -1
        while try_times >= 0:
            time.sleep(0.5)
            try:
                answer_middle = get_answer(
                    context,
                    data["history"],
                    data["documents"],
                    data["question"],
                    data["answer"],
                )
                answer_list = extract_numbers_from_string(answer_middle)
                # print(answer_list)
                if len(answer_list) == 0:
                    answer_final = -1
                else:
                    answer_final = answer_list[0]
                break
            except Exception as e:
                print("Error:", e)
                try_times -= 1
        result_list.append({"uuid": str(uuid), "quality": answer_final})

    data = pd.json_normalize(result_list)
    data.to_csv("answer_quality.csv", index=None)
