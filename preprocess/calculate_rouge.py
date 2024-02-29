import json
from rouge import Rouge
from tqdm import tqdm

rouge = Rouge()

with open("../multi_stage/idea_1/20240118-1000/release_train_data.json", "r") as f:
    train_data = json.load(f)

score_total = 0
score_num = 0


def lcs(s, t):
    len1 = len(s)
    len2 = len(t)
    res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if t[i - 1] == s[j - 1]:
                res[i][j] = 1 + res[i - 1][j - 1]
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    l = res[-1][-1]
    p = l / len1
    r = l / len2
    f = 2 * p * r / (p + r)
    return p, r, f


for data in tqdm(train_data[:100]):
    try:
        predict = data["predict"].lower()
        answer = data["answer"].lower()
        score = lcs(predict, answer)
        score_total += score
        score_num += 1
    except Exception as e:
        print(e)

print(score_total)
print(score_num)
print(score_total / score_num)
