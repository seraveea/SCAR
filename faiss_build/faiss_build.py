import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, CLIPTokenizer
from model.retriever import WikipediaKnowledgeBaseEntry


# ====== Load EVA-CLIP Model ======
model = AutoModel.from_pretrained(
    "hugging_face_cache/EVA-CLIP-8B",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = CLIPTokenizer.from_pretrained("hugging_face_cache/clip-vit-large-patch14")
model.to('cuda').eval()

# ====== Load KB Entries ======
with open("../vqa_data/infoseek_kb/wiki_100_dict_v4.json", "r") as f:
    kb_entries = json.load(f)


kb_base = [WikipediaKnowledgeBaseEntry(entry) for _, entry in kb_entries.items()]
print(f"[INFO] Total KB entries: {len(kb_base)}")

# ====== Init FAISS Index ======
embedding_dim = 1280  # for EVA-CLIP
index = faiss.IndexFlatIP(embedding_dim)

# ====== Batch Encode & Write into FAISS ======
def process_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
    with torch.no_grad():
        features = model.encode_text(inputs).float()
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    index.add(features.detach().cpu().numpy())
    del inputs, features
    torch.cuda.empty_cache()

# ====== Main Loop ======
batch_size = 1024
buffer_texts = []

for entry in tqdm(kb_base, desc="Building FAISS index"):
    buffer_texts.append(entry.title)
    if len(buffer_texts) == batch_size:
        process_batch(buffer_texts)
        buffer_texts = []

# Flush last batch
if buffer_texts:
    process_batch(buffer_texts)

# ====== Save FAISS index ======
faiss.write_index(index, "infoseek_title.index")
print(f"[INFO] Index built and saved: infoseek_title.index")