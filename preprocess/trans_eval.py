import json
from googletrans import Translator
from tqdm import tqdm
import time

with open("../../data/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

print(len(eval_data))


translator = Translator(service_urls=[
    'translate.google.com'
])

src_language = "en"
dst_language = "zh-cn"

def translate_own(translate_text):
    res = ""
    chunks = [translate_text[i:i +4000] for i in range(0, len(translate_text), 4000)]
    for c in chunks:
        res += translator.translate(c,src=src_language,dest=dst_language).text
    return res

eval_data_trans_sign_list =  [0 for i in range(len(eval_data))]

while 1:
    temp_index_list = []
    for index, num in enumerate(eval_data_trans_sign_list):
        if num == 0:
            temp_index_list.append(index)
    print(len(temp_index_list))
    if len(temp_index_list) != 0:
        for i, data in enumerate(tqdm(eval_data)):
            if i not in temp_index_list:
                continue
            time.sleep(5)
            try:
                history_len = len(data["history"])
                for j in range(history_len):
                    eval_data[i]["history"][j]["question"] = translate_own(eval_data[i]["history"][j]["question"])
                    eval_data[i]["history"][j]["answer"] = translate_own(eval_data[i]["history"][j]["answer"])

                document_list = []
                for d in data["documents"]:
                    document_list.append(translate_own(d))

                eval_data[i]["documents"] = document_list
                eval_data[i]["question"] = translate_own(eval_data[i]["question"])

                eval_data_trans_sign_list[i] = 1
            except Exception as e:
                print("error:",e)
    else:
        with open("../../data/release_phase1_eval_data_wo_gt_translate.json", "w") as f:
            json.dump(eval_data, f, ensure_ascii=False)
        break
