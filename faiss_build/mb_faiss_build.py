# build text and image faiss index using CLIP-EVA for mbier benchmark
# export CUDA_VISIBLE_DEVICES=5

import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, CLIPTokenizer, CLIPImageProcessor
import re
import PIL
def first_sentence(text):
    if not text:
        return ""
    text = text.strip()
    m = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
    return m[0]


# ====== Load EVA-CLIP Model ======
model = AutoModel.from_pretrained("hugging_face_cache/EVA-CLIP-8B", torch_dtype=torch.float16,trust_remote_code=True)
tokenizer = CLIPTokenizer.from_pretrained("hugging_face_cache/clip-vit-large-patch14")
processor = CLIPImageProcessor.from_pretrained("hugging_face_cache/clip-vit-large-patch14") # "openai/clip-vit-large-patch14"
model.to('cuda').eval()

# ====== Load KB Entries ======
# m-beir multimodal kb files


image_prefix = "vqa_data/M-BEIR/"

data_list = [json.loads(line) for line in open("vqa_data/M-BEIR/cand_pool/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl", encoding="utf-8")]
# ====== Init FAISS Index ======
embedding_dim = 1280  # for EVA-CLIP
index = faiss.IndexFlatIP(embedding_dim)

# ====== Batch Encode & Write into FAISS ======
def process_batch(texts, mode='text'):
    if mode == 'text':
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        with torch.no_grad():
            features = model.encode_text(inputs).float()
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        index.add(features.detach().cpu().numpy())
        del inputs, features
        # torch.cuda.empty_cache()
    else:
        for image in texts:
            inputs = (processor(images=image, return_tensors="pt").pixel_values.to("cuda").half())
            features = model.encode_image(inputs).float()
            features = torch.nn.functional.normalize(features)
            index.add(features.detach().cpu().numpy())
            
# ====== Main Loop ======

mode = 'image' # switch between text and image

if mode == 'text':
    batch_size = 1024
    buffer_texts = []
    for entry in tqdm(data_list, desc="Buidlding FAISS text index"):
        buffer_texts.append(entry['txt'])
        if len(buffer_texts) == batch_size:
            process_batch(buffer_texts, mode='text')
            buffer_texts = []
    # Flush last batch
    if buffer_texts:
        process_batch(buffer_texts)
elif mode == 'image':
    batch_size = 1
    buffer_texts = []
    for entry in tqdm(data_list, desc="Building FAISS image index"):
        image = PIL.Image.open(image_prefix+entry['img_path'])
        buffer_texts.append(image)
        if len(buffer_texts) == batch_size:
            process_batch(buffer_texts, mode='image')
            buffer_texts = []
    # Flush last batch
    if buffer_texts:
        process_batch(buffer_texts)

# ====== Save FAISS index ======
faiss.write_index(index, "vqa_data/MK2R/faiss_index/webqa_task2_image.index")
print(f"[INFO] Index built and saved")