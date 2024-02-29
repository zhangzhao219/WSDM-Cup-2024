from tqdm import tqdm
import openai
import json

context = "You are a helpful assistant."

context_user = "Now please answer the following [Question] briefly with our conversation above and the [Information] provided. Note that you should combine with the context of the conversation we had above, and you mustn't add other irrelevant information to your answer.\n"


def get_answer(context, history_list, document_list, question):
    history_json_list = []
    for history in history_list:
        history_json_list.append({"role": "user", "content": history["question"]})
        history_json_list.append({"role": "assistant", "content": history["answer"]})

    context_str = " "
    for doc in document_list:
        context_str += f"[Information] {doc}\n"

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
                "content": context_user + "[Question] " + question,
            },
        ]
    )

    response = gpt4(messages)
    res = response["choices"][0]["message"]["content"]
    # print(res)
    # exit()
    return res


if __name__ == "__main__":
    with open("../data/wsdm/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
        eval_data = json.load(f)

    # with open("../data/wsdm/ori/release_train_data.json", "r") as f:
    #     train_data = json.load(f)

    result_list = []
    eval_start_id = 12557

    for data in tqdm(eval_data):
        try_times = 10
        answer_final = ""
        while try_times >= 0:
            try:
                answer_final = get_answer(
                    context, data["history"], data["documents"], data["question"]
                )
                break
            except Exception as e:
                print("Error:", e)
                try_times -= 1
        result_list.append({"uuid": str(eval_start_id), "prediction": answer_final})
        eval_start_id += 1

    with open("answer_4.json", "w", encoding="utf-8") as writer:
        json.dump(result_list, writer, ensure_ascii=False, indent=4)
