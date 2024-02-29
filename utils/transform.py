import os
import sys
import math
import json
import copy
import shutil
import zipfile

file_name = ""
split_num = 2

if len(sys.argv) == 1:
    print(
        "Not specify a specific file, search infer_result_*.jsonl in the current folder..."
    )
    find_file = False
    for file in sorted(os.listdir("./")):
        if file.startswith("infer_result_") and file.endswith(".jsonl"):
            print(f"Find {file}")
            find_file = True
            file_name = file
            break
    if find_file == False:
        print("Not find a valid file!")
        exit()
elif len(sys.argv) == 2:
    file_name = sys.argv[1]
elif len(sys.argv) == 3:
    file_name = sys.argv[1]
    split_num = int(sys.argv[2])
else:
    print("The number of parameters is incorrect! Please check input!")
    exit()

with open(file_name, "r") as f:
    input_data = f.readlines()

final_data = []

uuid_list = [str(i) for i in range(14815,17939)] + ["0" + str(i) for i in range(4008,4472)]
# uuid_list = [str(i) for i in range(12557, 14351)]

for i, d in enumerate(input_data):
    temp_json = {}
    temp_json["uuid"] = uuid_list[i]
    temp_json["prediction"] = json.loads(d)["response"]
    final_data.append(temp_json)

file_name = file_name.split(".")[0]

if not os.path.isdir(file_name):
    os.mkdir(file_name)
else:
    print("Folder", file_name, "Exists, Delete it!")
    shutil.rmtree(file_name)
    os.mkdir(file_name)

every_num = math.floor(len(final_data) / split_num)
print("Every File has", str(every_num), "Data")

for each_num in range(split_num):
    num_start = each_num * every_num
    if each_num != split_num - 1:
        num_end = (each_num+1) * every_num
    else:
        num_end = len(final_data)
        if num_end - num_start != every_num:
            print("Last file has",str(num_end - num_start),"Data")

    t_data = copy.deepcopy(final_data)
    for i in range(len(final_data)):
        if i < num_start or i >= num_end:
            t_data[i]["prediction"] = ""
            
    with open(os.path.join(file_name, "submission.json"), "w") as f:
        json.dump(t_data, f, ensure_ascii=False, indent=4)

    zip_file = zipfile.ZipFile(os.path.join(file_name, str(uuid_list[num_start]) + "_" + str(uuid_list[num_end-1]) + ".zip"), "w")
    zip_file.write(
        filename=os.path.join(file_name, "submission.json"),
        arcname="submission.json",
        compress_type=zipfile.ZIP_DEFLATED,
    )
    zip_file.close()


with open(os.path.join(file_name, "submission.json"), "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

zip_file = zipfile.ZipFile(os.path.join(file_name, "submission.json.zip"), "w")
zip_file.write(
    filename=os.path.join(file_name, "submission.json"),
    arcname="submission.json",
    compress_type=zipfile.ZIP_DEFLATED,
)
zip_file.close()
