import os
import json
import numpy as np
import pandas as pd
from rouge import Rouge
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize 
from gensim.summarization import bm25

class BM25Retriever(object):
    def __init__(self, documents):
        self.documents = documents
        self.empty_sign = False
        self.bm25 = self._init_bm25()

    def _init_bm25(self):
        samples_for_retrieval_tokenized = []
        for document in self.documents:
            tokenized_example = word_tokenize(document)
            samples_for_retrieval_tokenized.append(tokenized_example)
        try:
            return bm25.BM25(samples_for_retrieval_tokenized)
        except:
            self.empty_sign = True
            return

    def compute_scores(self, query):
        if self.empty_sign:
            return [0 for i in range(len(self.documents))]
        tokenized_query = word_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        scores = []
        for idx in range(len(self.documents)):
            scores.append(bm25_scores[idx])
        return scores

rouge = Rouge()
rouge_list = ["p", "r", "f"]

embeddding_dir = "../embeddings"

embedding_dict = {}

for file in os.listdir(embeddding_dir):
    if file.endswith(".npy"):
        temp_emb = np.load(os.path.join(embeddding_dir, file), allow_pickle=True).item()
        print(len(temp_emb))
        embedding_dict.update(temp_emb)

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

with open("../data/wsdm/prepare/ori/release_phase1_eval_data_wo_gt.json", "r") as f:
    eval_data = json.load(f)

final_data = []

# -----------train data---------------

for i, data in enumerate(tqdm(train_data)):
    temp_json = {}

    uuid = data["uuid"]

    temp_json["uuid"] = uuid
    temp_json["data"] = "train"
    temp_json["type"] = "document"

    # embedding
    q_emb = embedding_dict[uuid + "-train-question"].reshape(1, -1)
    hqq_emb = embedding_dict[uuid + "-train-all-question"].reshape(1, -1)
    hqaq_emb = embedding_dict[uuid + "-train-all-question-and-answer"].reshape(1, -1)
    a_emb = embedding_dict[uuid + "-train-answer"].reshape(1, -1)

    # text
    all_history_question_answer = ""
    all_history_question = ""
    for i, h in enumerate(data["history"]):
        all_history_question_answer = all_history_question_answer + h["question"] + "\n"
        all_history_question = all_history_question + h["question"] + "\n"
        all_history_question_answer = all_history_question_answer + h["answer"] + "\n"

    q_text = data["question"]
    hqq_text = all_history_question + data["question"]
    hqaq_text = all_history_question_answer + data["question"]
    a_text = data["answer"]

    # bm25
    sparse_retriever = BM25Retriever(data["documents"])

    bm25_q = sparse_retriever.compute_scores(q_text)
    bm25_hqq = sparse_retriever.compute_scores(hqq_text)
    bm25_hqaq = sparse_retriever.compute_scores(hqaq_text)
    bm25_a = sparse_retriever.compute_scores(a_text)

    for j, document in enumerate(data["documents"]):
        document_json = temp_json.copy()
        document_json["order"] = j
        if len(document) == 0:
            final_data.append(document_json)
            continue

        d_emb = embedding_dict[uuid + "-train-documents-" + str(j)].reshape(1, -1)

        document_json["embedding_q"] = cosine_similarity(d_emb, q_emb)[0][0]
        document_json["embedding_hqq"] = cosine_similarity(d_emb, hqq_emb)[0][0]
        document_json["embedding_hqaq"] = cosine_similarity(d_emb, hqaq_emb)[0][0]
        document_json["embedding_a"] = cosine_similarity(d_emb, a_emb)[0][0]

        try:
            score = rouge.get_scores(
                hyps=[document], refs=[q_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_q_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_q_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_hqq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_hqq_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqaq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_hqaq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_hqaq_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[a_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_a_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_a_" + m] = 0
        
        document_json["bm25_q"] = bm25_q[j]
        document_json["bm25_hqq"] = bm25_hqq[j]
        document_json["bm25_hqaq"] = bm25_hqaq[j]
        document_json["bm25_a"] = bm25_a[j]

        final_data.append(document_json)

# -----------eval data---------------

for i, data in enumerate(tqdm(eval_data)):
    temp_json = {}

    uuid = data["uuid"]

    temp_json["uuid"] = uuid
    temp_json["data"] = "eval"
    temp_json["type"] = "document"

    # embedding
    q_emb = embedding_dict[uuid + "-eval-question"].reshape(1, -1)
    hqq_emb = embedding_dict[uuid + "-eval-all-question"].reshape(1, -1)
    hqaq_emb = embedding_dict[uuid + "-eval-all-question-and-answer"].reshape(1, -1)

    # text
    all_history_question_answer = ""
    all_history_question = ""
    for i, h in enumerate(data["history"]):
        all_history_question_answer = all_history_question_answer + h["question"] + "\n"
        all_history_question = all_history_question + h["question"] + "\n"
        all_history_question_answer = all_history_question_answer + h["answer"] + "\n"

    q_text = data["question"]
    hqq_text = all_history_question + data["question"]
    hqaq_text = all_history_question_answer + data["question"]

    # bm25
    sparse_retriever = BM25Retriever(data["documents"])

    bm25_q = sparse_retriever.compute_scores(q_text)
    bm25_hqq = sparse_retriever.compute_scores(hqq_text)
    bm25_hqaq = sparse_retriever.compute_scores(hqaq_text)

    for j, document in enumerate(data["documents"]):
        document_json = temp_json.copy()
        document_json["order"] = j
        if len(document) == 0:
            final_data.append(document_json)
            continue

        d_emb = embedding_dict[uuid + "-eval-documents-" + str(j)].reshape(1, -1)

        document_json["embedding_q"] = cosine_similarity(d_emb, q_emb)[0][0]
        document_json["embedding_hqq"] = cosine_similarity(d_emb, hqq_emb)[0][0]
        document_json["embedding_hqaq"] = cosine_similarity(d_emb, hqaq_emb)[0][0]

        try:
            score = rouge.get_scores(
                hyps=[document], refs=[q_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_q_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_q_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_hqq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_hqq_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqaq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_hqaq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_hqaq_" + m] = 0
        
        document_json["bm25_q"] = bm25_q[j]
        document_json["bm25_hqq"] = bm25_hqq[j]
        document_json["bm25_hqaq"] = bm25_hqaq[j]

        final_data.append(document_json)

df = pd.json_normalize(final_data)
df.to_csv("../data/wsdm/score.csv", index=None)
