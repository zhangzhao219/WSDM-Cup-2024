from tqdm import tqdm
import json
import tiktoken
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("../pretrained/upstage/SOLAR-10.7B-Instruct-v1.0")
# tokenizer.add_special_tokens({"pad_token": "<unk>"})


def get_answer(history_list, document_list, question, answer=" "):
    history_str = ""
    for history in history_list:
        history_str += f"{history['question']}\n"
        history_str += f"{history['answer']}\n"

    context_str = ""
    for doc in document_list:
        context_str += f"{doc}\n"

    # len(enc.encode(history_str)),
    # len(enc.encode(context_str)),
    # len(enc.encode(question)),
    # len(enc.encode(answer)),

    return (
        len(tokenizer(history_str, return_tensors="pt")["input_ids"][0]),
        len(tokenizer(context_str, return_tensors="pt")["input_ids"][0]),
        len(tokenizer(question, return_tensors="pt")["input_ids"][0]),
        len(tokenizer(answer, return_tensors="pt")["input_ids"][0]),
    )


if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")

    with open("../multi_stage/idea_2/20240118-1500/release_train_data.json", "r") as f:
        train_data = json.load(f)

    # with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    #     eval_data = json.load(f)

    history_num_list = []
    documents_num_list = []
    all_num_list = []
    answer_num_list = []

    for data in tqdm(train_data):
        history_num, documents_num, question_num, answer_num = get_answer(
            data["history"], data["documents"], data["question"], data["answer"]
        )
        history_num_list.append(history_num)
        documents_num_list.append(documents_num)
        all_num_list.append(history_num + documents_num + question_num)
        answer_num_list.append(answer_num)

    # for data in tqdm(eval_data):
    #     history_num, documents_num, question_num, answer_num = get_answer(
    #         data["history"], data["documents"], data["question"]
    #     )
    #     history_num_list.append(history_num)
    #     documents_num_list.append(documents_num)
    #     all_num_list.append(history_num + documents_num + question_num)
    #     answer_num_list.append(answer_num)

    history_num_list = sorted(history_num_list)
    print("history_num_list", history_num_list[0], history_num_list[-1])
    documents_num_list = sorted(documents_num_list)
    print("documents_num_list", documents_num_list[0], documents_num_list[-1])
    all_num_list = sorted(all_num_list)
    print("all_num_list", all_num_list[0], all_num_list[-1])
    answer_num_list = sorted(answer_num_list)
    print("answer_num_list", answer_num_list[0], answer_num_list[-1])

    # fig, ax = plt.subplots(2, 2, figsize=(24, 20), sharey=True)
    # ax = ax.ravel()

    # ax[0].plot(
    #     [i for i in range(len(history_num_list))],
    #     history_num_list,
    # )
    # ax[0].set_title("history_num_list")

    # ax[1].plot(
    #     [i for i in range(len(documents_num_list))],
    #     documents_num_list,
    # )
    # ax[1].set_title("documents_num_list")

    # ax[2].plot(
    #     [i for i in range(len(all_num_list))],
    #     all_num_list,
    # )
    # ax[2].set_title("all_num_list")

    # ax[3].plot(
    #     [i for i in range(len(answer_num_list))],
    #     answer_num_list,
    # )
    # ax[3].set_title("answer_num_list")

    # plt.savefig("Token Calculate.png", bbox_inches="tight")
