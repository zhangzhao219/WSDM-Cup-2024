import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
data = pd.read_csv("../../data/wsdm/score.csv").fillna(0)
data["position"] = 0

last_uuid = -1
now_pos = -1
for label, row in tqdm(data.iterrows(), total=data.shape[0]):
    if row["uuid"] == last_uuid:
        now_pos += 1
        data.at[label, "position"] = now_pos
    else:
        last_uuid = row["uuid"]
        now_pos = 0
print(data)

data.drop(['uuid', 'data', 'type', 'order'], axis=1, inplace=True)

for method in ["pearson", "spearman"]:
    plt.figure(figsize=(20,20))
    sns.heatmap(data.corr(method), vmax=1, vmin=-1, center=0, annot=True)
    plt.savefig(method + '.png')