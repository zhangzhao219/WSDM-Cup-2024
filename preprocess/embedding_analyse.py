import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

embeddding_dir = "../embeddings"

embedding_dict = {}

for file in os.listdir(embeddding_dir):
    if file.endswith(".npy"):
        temp_emb = np.load(os.path.join(embeddding_dir, file), allow_pickle=True).item()
        print(len(temp_emb))
        embedding_dict.update(temp_emb)

print(len(embedding_dict))

with open("../data/wsdm/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

with open("../data/wsdm/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

fig, ax = plt.subplots(2, 2, figsize=(30, 20), sharey=True)
ax = ax.ravel()

# -----------train data---------------
q_to_d_sim = []
d_to_h_sim = []

for data in tqdm(train_data):
    uuid = data["uuid"]
    document_len = len(data["documents"])
    q_sim = embedding_dict[uuid + "-train-question"].reshape(1, -1)
    if len(data["history"]) != 0:
        h_sim = embedding_dict[uuid + "-train-history-question-and-answer"].reshape(
            1, -1
        )
    else:
        h_sim = 0
    for i in range(document_len):
        if len(data["documents"][i]) != 0:
            d_sim = embedding_dict[uuid + "-train-documents-" + str(i)].reshape(1, -1)
        else:
            d_sim = 0

        if type(d_sim) == int:
            q_to_d_sim.append(0)
            d_to_h_sim.append(0)
        elif type(h_sim) == int:
            d_to_h_sim.append(0)
            q_to_d_sim.append(cosine_similarity(d_sim, q_sim)[0][0])
        else:
            if cosine_similarity(d_sim, q_sim)[0][0] >0.85:
                print(uuid, i, "q_to_d_sim", cosine_similarity(d_sim, q_sim)[0][0])
                exit()
            q_to_d_sim.append(cosine_similarity(d_sim, q_sim)[0][0])
            d_to_h_sim.append(cosine_similarity(d_sim, h_sim)[0][0])

q_to_d_sim = sorted(q_to_d_sim)
d_to_h_sim = sorted(d_to_h_sim)

ax[0].plot(
    [i for i in range(len(q_to_d_sim))],
    q_to_d_sim,
)
ax[0].set_title("train: q_to_d_sim")

ax[1].plot(
    [i for i in range(len(d_to_h_sim))],
    d_to_h_sim,
)
ax[1].set_title("train: d_to_h_sim")

# -----------eval data---------------

q_to_d_sim = []
d_to_h_sim = []

for data in tqdm(eval_data):
    uuid = data["uuid"]
    if uuid != "12595":
        continue
    document_len = len(data["documents"])
    q_sim = embedding_dict[uuid + "-eval-question"].reshape(1, -1)
    if len(data["history"]) != 0:
        h_sim = embedding_dict[uuid + "-eval-history-question-and-answer"].reshape(
            1, -1
        )
    else:
        h_sim = 0
    for i in range(document_len):
        if len(data["documents"][i]) != 0:
            d_sim = embedding_dict[uuid + "-eval-documents-" + str(i)].reshape(1, -1)
        else:
            d_sim = 0

        if type(d_sim) == int:
            q_to_d_sim.append(0)
            d_to_h_sim.append(0)
        elif type(h_sim) == int:
            d_to_h_sim.append(0)
            # print(i, "q_to_d_sim", cosine_similarity(d_sim, q_sim)[0][0])
            q_to_d_sim.append(cosine_similarity(d_sim, q_sim)[0][0])
        else:
            # print(i, "q_to_d_sim", cosine_similarity(d_sim, q_sim)[0][0])
            # print(i, "d_to_h_sim", cosine_similarity(d_sim, h_sim)[0][0])
            q_to_d_sim.append(cosine_similarity(d_sim, q_sim)[0][0])
            d_to_h_sim.append(cosine_similarity(d_sim, h_sim)[0][0])

q_to_d_sim = sorted(q_to_d_sim)
d_to_h_sim = sorted(d_to_h_sim)

ax[2].plot(
    [i for i in range(len(q_to_d_sim))],
    q_to_d_sim,
)
ax[2].set_title("eval: q_to_d_sim")

ax[3].plot(
    [i for i in range(len(d_to_h_sim))],
    d_to_h_sim,
)
ax[3].set_title("eval: d_to_h_sim")

plt.savefig("Similarity.png", bbox_inches="tight")
