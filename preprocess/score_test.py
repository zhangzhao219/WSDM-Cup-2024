import os
import json
import numpy as np
import pandas as pd
from rouge import Rouge
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize 
from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer

sentence_transformer_model = SentenceTransformer("../pretrained/nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

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

with open("../data/wsdm/prepare/ori/release_train_data.json", "r") as f:
    train_data = json.load(f)

with open("../data/wsdm/prepare/ori/release_phase2_test_data_wo_gt.json", "r") as f:
    test_data = json.load(f)

final_data = []

# -----------train data---------------

for i, data in enumerate(tqdm(train_data)):
    temp_json = {}

    uuid = data["uuid"]

    temp_json["uuid"] = uuid
    temp_json["data"] = "train"
    temp_json["type"] = "document"

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

    # embedding
    q_emb = sentence_transformer_model.encode(q_text).reshape(1, -1)
    hqq_emb = sentence_transformer_model.encode(hqq_text).reshape(1, -1)
    hqaq_emb = sentence_transformer_model.encode(hqaq_text).reshape(1, -1)
    a_emb = sentence_transformer_model.encode(a_text).reshape(1, -1)
    d_emb_all = sentence_transformer_model.encode(data["documents"])

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

        d_emb = d_emb_all[j].reshape(1, -1)

        document_json["embedding_q"] = cosine_similarity(d_emb, q_emb)[0][0]
        document_json["embedding_hqq"] = cosine_similarity(d_emb, hqq_emb)[0][0]
        document_json["embedding_hqaq"] = cosine_similarity(d_emb, hqaq_emb)[0][0]
        document_json["embedding_a"] = cosine_similarity(d_emb, a_emb)[0][0]

        try:
            score = rouge.get_scores(
                hyps=[document], refs=[q_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_q_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_q_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_hqq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_hqq_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqaq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_hqaq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_hqaq_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[a_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_a_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_a_" + m] = 0

        try:
            score = lcs(document, q_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_q_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_q_" + n] = 0
        try:
            score = lcs(document, hqq_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqq_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqq_" + n] = 0
        try:
            score = lcs(document, hqaq_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqaq_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqaq_" + n] = 0
        try:
            score = lcs(document, a_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_a_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_a_" + n] = 0
        
        document_json["bm25_q"] = bm25_q[j]
        document_json["bm25_hqq"] = bm25_hqq[j]
        document_json["bm25_hqaq"] = bm25_hqaq[j]
        document_json["bm25_a"] = bm25_a[j]

        final_data.append(document_json)

# -----------test data---------------

for i, data in enumerate(tqdm(test_data)):
    temp_json = {}

    uuid = data["uuid"]

    temp_json["uuid"] = uuid
    temp_json["data"] = "test"
    temp_json["type"] = "document"

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

    # embedding
    q_emb = sentence_transformer_model.encode(q_text).reshape(1, -1)
    hqq_emb = sentence_transformer_model.encode(hqq_text).reshape(1, -1)
    hqaq_emb = sentence_transformer_model.encode(hqaq_text).reshape(1, -1)
    d_emb_all = sentence_transformer_model.encode(data["documents"])

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

        d_emb = d_emb_all[j].reshape(1, -1)

        document_json["embedding_q"] = cosine_similarity(d_emb, q_emb)[0][0]
        document_json["embedding_hqq"] = cosine_similarity(d_emb, hqq_emb)[0][0]
        document_json["embedding_hqaq"] = cosine_similarity(d_emb, hqaq_emb)[0][0]

        try:
            score = rouge.get_scores(
                hyps=[document], refs=[q_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_q_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_q_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_hqq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_hqq_" + m] = 0
        try:
            score = rouge.get_scores(
                hyps=[document], refs=[hqaq_text], avg=True
            )["rouge-l"]
            for m in rouge_list:
                document_json["rouge_word_hqaq_" + m] = score[m]
        except:
            print(uuid)
            for m in rouge_list:
                document_json["rouge_word_hqaq_" + m] = 0

        try:
            score = lcs(document, q_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_q_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_q_" + n] = 0
        try:
            score = lcs(document, hqq_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqq_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqq_" + n] = 0
        try:
            score = lcs(document, hqaq_text)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqaq_" + n] = score[m]
        except:
            print(uuid)
            for m,n in enumerate(rouge_list):
                document_json["rouge_character_hqaq_" + n] = 0
        
        document_json["bm25_q"] = bm25_q[j]
        document_json["bm25_hqq"] = bm25_hqq[j]
        document_json["bm25_hqaq"] = bm25_hqaq[j]

        final_data.append(document_json)

df = pd.json_normalize(final_data)
df.to_csv("../data/wsdm/score_train_test.csv", index=None)
