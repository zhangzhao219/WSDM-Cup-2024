from tqdm import tqdm
import json


def get_answer(history_list, document_list, question):
    history_str = ""
    for history in history_list:
        history_str += f"{history['question']}\n"
        history_str += f"{history['answer']}\n"

    context_str = ""
    for doc in document_list:
        context_str += f"{doc}\n"

    return history_str + "\n" + context_str + "\n" + question


if __name__ == "__main__":
    with open("../data/wsdm/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
        eval_data = json.load(f)

    result_list = []
    eval_start_id = 12557

    for data in tqdm(eval_data):
        result_list.append(
            {
                "uuid": str(eval_start_id),
                "prediction": get_answer(
                    data["history"], data["documents"], data["question"]
                ),
            }
        )
        eval_start_id += 1

    with open("all_answer.json", "w", encoding="utf-8") as writer:
        json.dump(result_list, writer, ensure_ascii=False, indent=4)
