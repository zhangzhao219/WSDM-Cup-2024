import sys
import json

input_file = sys.argv[1]

number_json_file = "../data/wsdm/model/eval_split_document/eval_num_order.json"

output_file = sys.argv[2]

with open(input_file, "r") as f:
    input_data = f.readlines()

with open(number_json_file, "r") as f:
    number_count_dict = json.load(f)


final_data = []


now_index = 0

for key,num in number_count_dict.items():
    temp_json = {}
    temp_json["uuid"] = key
    num_len = len(num)
    prediction_str = ""
    for i in range(num_len):
        prediction_str = prediction_str + json.loads(input_data[now_index+i])["response"] + "\n"
    temp_json["prediction"] = prediction_str
    final_data.append(temp_json)
    now_index += num_len

with open(output_file, "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
