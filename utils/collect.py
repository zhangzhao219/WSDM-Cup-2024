import os
import shutil
from datetime import datetime


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


datetime_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(datetime_now)

shutil.copy("transform.py", os.path.join(datetime_now, "transform.py"))

find_dir_list = [
    "/home/aiscuser/Swift-Scripts/merge/data/test/20240212",
]

for root_dir in find_dir_list:
    for file in findAllFile(root_dir):
        file_part = file.split("/")
        if file_part[-1].startswith("infer_result_") and file_part[-1].endswith(
            ".jsonl"
        ):
            file_name = (
                "infer_result_"
                + file_part[-3]
                + "-"
                + file_part[-2].split("-")[1]
                + "-"
                + file_part[-1]
            )
            shutil.copy(
                file,
                os.path.join(datetime_now, file_name),
            )
            os.system("python " + datetime_now + "/transform.py " + datetime_now + "/" + file_name)
            os.remove(os.path.join(datetime_now, file_name))

os.remove(os.path.join(datetime_now, "transform.py"))
