from swift.tuners import SwiftModel
from transformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

BASE_MODEL_PATH = "/home/aiscuser/Swift-Scripts/pretrained/upstage/SOLAR-10.7B-Instruct-v1.0"
LORA_ADAPTER_PATH = "/home/aiscuser/Swift-Scripts/submit/checkpoints/v13-20240202-072530"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="cpu")
print(sum(p.numel() for p in base_model.parameters()))

model = SwiftModel.from_pretrained(base_model, LORA_ADAPTER_PATH, device_map="cpu")
print(sum(p.numel() for p in model.parameters()))

embedding_model = SentenceTransformer("../pretrained/nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
print(sum(p.numel() for p in embedding_model.parameters()))
