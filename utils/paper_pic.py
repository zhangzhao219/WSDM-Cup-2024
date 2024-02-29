import matplotlib.pyplot as plt
print(plt.style.available)

# plt.style.use('seaborn-v0_8-pastel')
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax = ax.ravel()

labels = ['Word-level ROUGE-L', 'Character-level ROUGE-L', 'Keywords Recall']
char_a_f = [0.4631253770, 0.620643312, 0.6858702288]
emb_a_s = [0.4640529782, 0.6212052121, 0.68974889]
word_a_f = [0.4636185277,0.619910912, 0.687988400]

char_a_f[0] = char_a_f[0] + 0.1
emb_a_s[0] = emb_a_s[0] + 0.1
word_a_f[0] = word_a_f[0] + 0.1

x = [0,1,2]
width = 0.2

ax[0].bar([i - width-0.1 for i in x], char_a_f, width, label="char_a_f", alpha=0.8)
ax[0].bar([i for i in x], emb_a_s, width, label="emb_a_s", alpha=0.8)
ax[0].bar([i + width+0.1 for i in x], word_a_f, width, label="word_a_f", alpha=0.8)
ax[0].set_xticks(x, labels=labels)
# ax[0].set_xlabel("Embedding Methods")
ax[0].set_ylim(0.55, 0.7)
ax[0].set_yticks([0.55, 0.6, 0.65, 0.7], labels=[0.45, 0.6, 0.65, 0.7])
ax[0].set_ylabel("Metrics")
ax[0].set_title("(a)")
ax[0].legend()

num = [3,4,5,6,7,8]
emb_s_wr = [0.461165145, 0.4620666179, 0.4619641564, 0.4631399327, 0.4637296673, 0.4653601418]
emb_s_cr = [0.6182329204, 0.6186608208, 0.6189379213, 0.6195481089, 0.620055073, 0.62083712097]
emb_s_kr = [0.6892007750, 0.689883606, 0.6934756065, 0.6931504485, 0.6943070818, 0.6953475871]

emb_s_wr = [i + 0.1 for i in emb_s_wr]

ax[1].scatter(2, 0.4554037598343195+0.1, marker='*')
ax[1].scatter(2, 0.6143637937814926, marker='*')
ax[1].scatter(2, 0.6813458884111055, marker='*')
ax[1].plot(num, emb_s_wr, label="Word-level ROUGE-L", marker='o')
ax[1].plot(num, emb_s_cr, label="Character-level ROUGE-L", marker='o')
ax[1].plot(num, emb_s_kr, label="Keywords Recall", marker='o')
ax[1].set_yticks([])
ax[1].set_xlabel("Number of models")
ax[1].set_xticks([2]+num, labels=[1]+num)
ax[1].set_title("(b)")
ax[1].legend(loc=(0.55, 0.65))

fig.tight_layout()
fig.savefig("Ensemble.png")