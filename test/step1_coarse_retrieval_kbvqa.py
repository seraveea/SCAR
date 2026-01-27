from argparse import ArgumentParser
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates, semantic_entropy, EntailmentDeberta, query_image_loader, map_id2title
import PIL
import os
from model import ClipRetriever, SCAR

INDEX_PATH_EVQA = {
    "summary":"../vqa_data/evqa_summary_llama3.index",
    "title":"../vqa_data/evqa_title.index",
    "image":"../vqa_data/evqa_image_index/"
}
INDEX_PATH_INFO = {
    "summary":"../vqa_data/infoseek_summary_llama3.index",
    "title":"../vqa_data/infoseek_title.index",
    "image":"../vqa_data/infoseek_image_index/"
}

def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall

def zscore_normalize(results, key="similarity"):
    scores = np.array([e[key].detach().cpu() for e in results], dtype=float)
    mean, std = scores.mean(), scores.std()

    for e in results:
        if std == 0:
            e[key] = 0.0 
        else:
            e[key] = (e[key] - mean) / std
    return results

def minmax(scores):
    scores = np.array(scores, dtype=float)
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)

def generate_entry_sim(top_k, retrieval_top_k):
    entry = remove_list_duplicates([retrieved_entry["kb_entry"] for retrieved_entry in top_k])
    seen = set()
    sim = [top_k[i]["similarity"] for i in range(retrieval_top_k) if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))]
    return entry, sim

def run_retrieval(sample_file_path: str, knowledge_base_path: str, faiss_index_path: str,
    top_ks: list, retrieval_top_k: int, **kwargs):
    sample_list, sample_header = load_csv_data(sample_file_path)
    image_loader = query_image_loader()
    time_counter = []
    retriever = ClipRetriever(device="cuda", model=kwargs["retriever_vit"])
    print("Knowledge Base Loading")
    knowledge_base_list = retriever.load_knowledge_base(knowledge_base_path)
    if kwargs["index_modal"] == 'summary':
        retriever.load_summary_faiss_index(faiss_index_path)
    elif kwargs["index_modal"] == 'title':
        retriever.load_title_faiss_index(faiss_index_path)
    elif kwargs["index_modal"] == 'image':
        retriever.load_image_faiss_index(faiss_index_path)
    elif kwargs["index_modal"] == 'ours':
        if 'evqa' in sample_file_path:
            retriever.load_summary_faiss_index(INDEX_PATH_EVQA['summary'])
            retriever.load_title_faiss_index(INDEX_PATH_EVQA['title'])
            retriever.load_image_faiss_index(INDEX_PATH_EVQA['image'])
            image_embedding = np.load('../vqa_data/evqa_image_embedding.npy')
            title_embedding = np.load('../vqa_data/evqa_title_embedding.npy')
            summa_embedding = np.load('../vqa_data/evqa_summary_embedding.npy')
        elif 'infoseek' in sample_file_path:
            retriever.load_summary_faiss_index(INDEX_PATH_INFO['summary'])
            retriever.load_title_faiss_index(INDEX_PATH_INFO['title'])
            retriever.load_image_faiss_index(INDEX_PATH_INFO['image'])
            image_embedding = np.load('../vqa_data/infoseek_image.npy')
            title_embedding = np.load('../vqa_data/infoseek_title.npy')
            summa_embedding = np.load('../vqa_data/infoseek_summary.npy')
        mr_tool = SCAR()
    print("Knowledge Base Loaded")

    recalls = {k: 0 for k in top_ks}
    retrieval_result = {}
    # ------split all questions into several parts------------
    element_local_index = [i for i in range(len(sample_list))]
    if kwargs['total']:
        part_num = len(sample_list)//kwargs['total']
        element_local_index = element_local_index[(kwargs['split']-1)*part_num:kwargs['split']*part_num]

    for it in tqdm(element_local_index):
        # -----------timer----------------
        t0 = time.perf_counter()
        # -------------------------------------
        example = get_test_question(it, sample_list, sample_header)
        ground_truth = example["wikipedia_url"] # the position of the correct answer
        # ---------------load image----------------
        if example["dataset_name"] == "infoseek":
            image = PIL.Image.open(image_loader.get_test_image(example["data_id"], example["dataset_name"]))
        else:
            image = PIL.Image.open(image_loader.get_test_image(example["dataset_image_ids"].split("|")[0],example["dataset_name"]) )
        # ---------------create id------------------
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)
        # ----------------describe image, beam search----------
        query_input = {'image':image}
        # ----------------embed query_input---------
        query_emb_dict = retriever.Encoding_Query_Input(query_input)
        query_emb = query_emb_dict['img_query_emb']
        # ----------------coarse-grained retrieval------------
        if kwargs["index_modal"] == 'summary':
            top_k = retriever.retrieval_summary_faiss(query_emb, top_k=retrieval_top_k)
        elif kwargs["index_modal"] == 'title':
            top_k = retriever.retrieval_title_faiss(query_emb, top_k=retrieval_top_k)
        elif kwargs["index_modal"] == 'image':
            # using image retrieval may result in multiple images corresponding to one entity, here only get retrieval_top_k, no overshot setting, if always retrieve the same one, it also shows this one is really close
            top_k = retriever.retrieval_image_faiss(query_emb, top_k=retrieval_top_k)
        elif kwargs["index_modal"] == 'ours':
            img_query_emb = query_emb_dict['img_query_emb']
            top_k = retriever.retrieval_ours(img_query_emb, mr_tool, image_embedding, title_embedding, summa_embedding)

        
        # --------------organize retrieval results------------
        top_k_wiki = [retrieved_entry["url"] for retrieved_entry in top_k]
        top_k_wiki = remove_list_duplicates(top_k_wiki)
        # ----------------store retrieval results---------
        if kwargs["save_result_path"] != "None":
            entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
            entries = remove_list_duplicates(entries)
            seen = set()
            retrieval_simlarities = [top_k[i]["similarity"] for i in range(min(retrieval_top_k, len(top_k_wiki))) if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))]
            retrieval_result[data_id] = {
                "question": example['question'],
                "image": example['data_id'] if example["dataset_name"] == "infoseek" else example["dataset_image_ids"],
                "ground_truth": ground_truth,
                "retrieved_entities": [{'url': entry.url, 'title': entry.title} for entry in entries[:20]],
                "retrieval_similarities": [float(sim) for sim in retrieval_simlarities[:20]],
            }
        # ---------------calculate recall-----------------
        recall = eval_recall(top_k_wiki, ground_truth, top_ks)
        for k in top_ks:
            recalls[k] += recall[k]


        time_counter.append(time.perf_counter()-t0)

     # ----------store coarse-grained retrieval results and calculate recall----------
    for k in top_ks:
        print("Avg Recall@{}: ".format(k), recalls[k] / (it+1))
    
    print(f"Avg coarse-grained retrieval time per query: {np.mean(time_counter)*1000:.4f} ± {np.std(time_counter)*1000:.4f} ms")

    if kwargs["save_result_path"] != "None":
        os.makedirs(os.path.dirname(kwargs["save_result_path"]), exist_ok=True)
        print("Save retrieval result to: ", kwargs["save_result_path"])
        with open(kwargs["save_result_path"], "w") as f:
            json.dump(retrieval_result, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--index_modal", type=str, required=True, help="choose one from summary,image, title or ours")
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument("--top_ks", type=str, default="1,5,10,20,100", help="comma separated list of top k values, e.g. 1,5,10,20,100",)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument( "--retriever_vit", type=str, default="clip", help="clip or eva-clip")
    parser.add_argument("--save_result_path", type=str, default="None", help="path to save retrieval result")
    parser.add_argument("--split",type=int, default=None, help='split all into several splits')
    parser.add_argument("--total",type=int, default=None, help='the total number of splits')
    
    args = parser.parse_args()
    if args.total:
        postfix = str(args.split)+'-'+str(args.total)
        args.save_result_path = args.save_result_path + postfix
    retrieval_config = {
        "sample_file_path": args.sample_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "retriever_vit": args.retriever_vit,
        "save_result_path": args.save_result_path,
        "index_modal": args.index_modal,
        "split": args.split,
        "total": args.total,
    }
    print("------------------------------------")
    print("retrieval_config: ", retrieval_config)
    print("------------------------------------")

    run_retrieval(**retrieval_config)
