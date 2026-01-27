import faiss
import numpy as np

index = faiss.read_index('../vqa_data/M-BEIR/faiss_index/oven_task8_image.index')

n = index.ntotal
d = index.d

xb = np.zeros((n, d), dtype=np.float32)
index.reconstruct_n(0, n, xb)

np.save('../vqa_data/M-BEIR/faiss_index/oven_task8_image.npy', xb)

print(f"Saved {xb.shape} vectors to npy")