from tqdm import tqdm
import openai
import json
import time

context = "Here's a list of questions and answers that users asked (the last question hasn't been answered yet) and information about the context needed to answer the last question. Now you are a helpful assistant, if you must answer to the last question, please give no more than eight keywords that you think must be included in the answer. Note that you don't need to answer the question, just give keywords only."


def get_answer(context, history_list, document_list, question):
    history_str = ""
    for history in history_list:
        history_str += f"[Question] {history['question']}\n"
        history_str += f"[Answer] {history['answer']}\n"

    context_str = ""
    for doc in document_list:
        context_str += f"[Context] {doc}\n"

    messages = [
        {"role": "system", "content": context},
        {
            "role": "user",
            "content": history_str + context_str + "[Question] " + question + "\n",
        },
    ]
    # print(messages[1]["content"])
    response = gpt4(messages)
    res = response["choices"][0]["message"]["content"]
    # print(res)
    return res


if __name__ == "__main__":

    with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
        train_data = json.load(f)

    with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
        eval_data = json.load(f)

    result_list = []

    for data in tqdm(eval_data):
        time.sleep(1)
        try_times = 10
        keywords = " "
        while try_times >= 0:
            try:
                keywords = get_answer(
                    context, data["history"], data["documents"], data["question"]
                )
                break
            except Exception as e:
                print("Error:", e)
                try_times -= 1
        result_list.append({"uuid": data["uuid"], "prediction": keywords})

    with open("4_keyword_eval.json", "w", encoding="utf-8") as writer:
        json.dump(result_list, writer, ensure_ascii=False, indent=4)
