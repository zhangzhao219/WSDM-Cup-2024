import json
from googletrans import Translator
from tqdm import tqdm
import time

with open("../../data/release_train_data.json", "r") as f:
    train_data = json.load(f)

print(len(train_data))


translator = Translator(service_urls=["translate.google.com"])

src_language = "en"
dst_language = "zh-cn"


def translate_own(translate_text):
    res = ""
    chunks = [translate_text[i : i + 4000] for i in range(0, len(translate_text), 4000)]
    for c in chunks:
        res += translator.translate(c, src=src_language, dest=dst_language).text
    return res


train_data_trans_sign_list = [0 for i in range(len(train_data))]

for z in range(100):
    temp_index_list = []
    for index, num in enumerate(train_data_trans_sign_list):
        if num == 0:
            temp_index_list.append(index)
    print(len(temp_index_list))
    if len(temp_index_list) != 0:
        for i, data in enumerate(tqdm(train_data)):
            if i not in temp_index_list:
                continue
            time.sleep(8)
            try:
                history_len = len(data["history"])
                for j in range(history_len):
                    train_data[i]["history"][j]["question"] = translate_own(
                        train_data[i]["history"][j]["question"]
                    )
                    train_data[i]["history"][j]["answer"] = translate_own(
                        train_data[i]["history"][j]["answer"]
                    )

                document_list = []
                for d in data["documents"]:
                    document_list.append(translate_own(d))

                train_data[i]["documents"] = document_list
                train_data[i]["question"] = translate_own(train_data[i]["question"])
                train_data[i]["answer"] = translate_own(train_data[i]["answer"])

                train_data_trans_sign_list[i] = 1
            except Exception as e:
                print("error:", e)
    else:
        with open("../data/wsdm/translate/release_train_data_translate.json", "w") as f:
            json.dump(train_data, f, ensure_ascii=False)
        exit()
with open("../data/wsdm/translate/release_train_data_translate.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False)