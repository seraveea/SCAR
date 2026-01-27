from argparse import ArgumentParser
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import remove_list_duplicates
import PIL
import os
import csv
from model import MBEIRClipRetriever, MR_GR


IMAGE_PREFIX = '/data/user/seraveea/research/vqa_data/M-BEIR/'

DATASET_MAP = {
    "WebQA": "../vqa_data/M-BEIR/query/query/test/mbeir_webqa_task2_test.jsonl",
    "OVEN": "../vqa_data/M-BEIR/query/query/test/mbeir_oven_task8_test.jsonl"
}
FAISS_MAP = {
    "WebQA_text": '../vqa_data/M-BEIR/faiss_index/webqa_task2_text.index',
    "WebQA_image": '../vqa_data/M-BEIR/faiss_index/webqa_task2_image.index',
    "OVEN_text": '../vqa_data/M-BEIR/faiss_index/oven_task8_text.index',
    "OVEN_image": '../vqa_data/M-BEIR/faiss_index/oven_task8_image.index',
    "WebQA_ours": None,
    "OVEN_ours": None,
}
KNOWLEDGE_MAP = {
    "WebQA": '../vqa_data/M-BEIR/cand_pool/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl',
    "OVEN": '../vqa_data/M-BEIR/cand_pool/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl'
}


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_retrieval(sample_file_path: str, knowledge_base_path: str, faiss_index_path: str,
    top_ks: list, retrieval_top_k: int, **kwargs):
    sample_list = [json.loads(line) for line in open(sample_file_path, encoding="utf-8")]
    retriever = MBEIRClipRetriever(device="cuda", model=kwargs["retriever_vit"]) # retriever
    knowledge_base_list = retriever.load_knowledge_base(knowledge_base_path) # knowledge path
    if kwargs["index_modal"] == 'text':
        retriever.load_text_faiss_index(faiss_index_path)
    elif kwargs["index_modal"] == 'image':
        retriever.load_image_faiss_index(faiss_index_path)
    elif kwargs["index_modal"] == 'ours':
        if kwargs['dataset'] == 'WebQA':
            retriever.load_text_faiss_index(FAISS_MAP['WebQA_text'])
            retriever.load_image_faiss_index(FAISS_MAP['WebQA_image'])
            title_embedding = np.load('../vqa_data/M-BEIR/faiss_index/webqa_task2_text.npy')
            image_embedding = np.load('../vqa_data/M-BEIR/faiss_index/webqa_task2_image.npy')
        elif kwargs['dataset'] == 'OVEN':
            retriever.load_text_faiss_index(FAISS_MAP['OVEN_text'])
            retriever.load_image_faiss_index(FAISS_MAP['OVEN_image'])
            title_embedding = np.load('../vqa_data/M-BEIR/faiss_index/oven_task8_text.npy')
            image_embedding = np.load('../vqa_data/M-BEIR/faiss_index/oven_task8_image.npy')
        mr_tool = MR_GR(priority=(1,0))
    print("Knowledge Base Loaded")

    retrieval_result = {}
    element_local_index = [i for i in range(len(sample_list))]
    for it in tqdm(element_local_index):
        example = sample_list[it]
        ground_truth = example["pos_cand_list"]
        data_id = example["qid"]
        query = example['query_txt']
        if kwargs['dataset'] == 'OVEN':
            image_path = IMAGE_PREFIX+example['query_img_path']
            image = PIL.Image.open(image_path)
            query_input = {'image':image}
            query_emb_dict = retriever.Encoding_Query_Input(query_input)
            query_emb = query_emb_dict['img_query_emb']
        else:
            query_input = {'captions':query}
            query_emb_dict = retriever.Encoding_Query_Input(query_input)
            query_emb = query_emb_dict['cap_query_emb']

        if kwargs["index_modal"] == 'text':
            top_k = retriever.retrieval_text_faiss(query_emb, top_k=retrieval_top_k)
        elif kwargs["index_modal"] == 'image':
            top_k = retriever.retrieval_image_faiss(query_emb, top_k=retrieval_top_k)
        elif kwargs["index_modal"] == 'ours':
            top_k, raw = retriever.retrieval_ours(query_emb, mr_tool, title_embedding, image_embedding)

        top_k_wiki = [retrieved_entry["id"] for retrieved_entry in top_k]
        top_k_wiki = remove_list_duplicates(top_k_wiki)

        if kwargs["save_result_path"] != "None":
            entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
            seen = set()
            retrieval_simlarities = [top_k[i]["similarity"] for i in range(min(retrieval_top_k, len(top_k_wiki))) if not (top_k[i]["id"] in seen or seen.add(top_k[i]["id"]))]
            retrieval_result[data_id] = {
                "question_id": data_id,
                "ground_truth": ground_truth,
                "retrieved_entities": [{'id': entry['did']} for entry in entries[:20]],
                "retrieval_similarities": [float(sim) for sim in retrieval_simlarities[:20]],

            }
    if kwargs["save_result_path"] != "None":
        os.makedirs(os.path.dirname(kwargs["save_result_path"]), exist_ok=True)
        print("Save retrieval result to: ", kwargs["save_result_path"])
        with open(kwargs["save_result_path"], "w") as f:
            json.dump(retrieval_result, f, indent=4, default=int)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="WebQA,OVEN")
    parser.add_argument("--index_modal", type=str, required=True, help="choose one from text, image or ours")
    parser.add_argument("--top_ks", type=str, default="1,5,10,20,100", help="comma separated list of top k values, e.g. 1,5,10,20,100",)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument( "--retriever_vit", type=str, default="clip", help="clip or eva-clip")
    parser.add_argument("--save_result_path", type=str, default="None", help="path to save retrieval result")
    
    args = parser.parse_args()

    sample_file = DATASET_MAP[args.dataset]
    faiss_path = FAISS_MAP[args.dataset+"_"+args.index_modal]
    knowldege_base = KNOWLEDGE_MAP[args.dataset]
    retrieval_config = {
        "dataset": args.dataset,
        "sample_file_path": sample_file,
        "faiss_index_path": faiss_path,
        "knowledge_base_path": knowldege_base,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "retriever_vit": args.retriever_vit,
        "save_result_path": args.save_result_path,
        "index_modal": args.index_modal,
    }
    print("------------------------------------")
    print("retrieval_config: ", retrieval_config)
    print("------------------------------------")

    run_retrieval(**retrieval_config)