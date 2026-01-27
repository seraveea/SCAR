"""  
Serves as the initial-search retriever
"""
import os
import torch
import tqdm
import pickle
import json
from transformers import AutoModel, AutoProcessor, CLIPTokenizer, CLIPImageProcessor
import faiss
import numpy as np
from faiss import write_index, read_index
import faiss.contrib.torch_utils
from pathlib import Path
import sys
QWEN3_ROOT = Path("/data/user/seraveea/research/Qwen3-VL-Embedding")
sys.path.insert(0, str(QWEN3_ROOT))
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder


INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

class KnowledgeBase:
    """Knowledge base for OMGM system.

    Returns:
        KnowledgeBase
    """

    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)

    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None

    def load_knowledge_base(self):
        """Load the knowledge base."""
        raise NotImplementedError

class WikipediaKnowledgeBase(KnowledgeBase):
    """Knowledge base for OMGM."""

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        super().__init__(knowledge_base_path)
        self.knowledge_base = []

    def load_knowledge_base_full(
        self, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base from multiple score files.
        ‘

        Args:
            image_dict: The image dictionary to load.
            scores_path: The parent folder path to the vision similarity scores to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None

        if visual_attr is not None:
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if scores_path is not None:
            # get the image scores for each entry
            # get all the *.pkl files in the scores_path
            print("Loading knowledge base score from {}.".format(scores_path))
            import glob

            score_files = glob.glob(scores_path + "/*.pkl")
            image_scores = {}
            for score_file in tqdm.tqdm(score_files):
                try:
                    with open(score_file, "rb") as f:
                        image_scores.update(pickle.load(f))
                except:
                    raise FileNotFoundError(
                        "Image scores not found, which should be a url or path to a pickle file."
                    )
            print("Loaded {} image scores.".format(len(image_scores)))
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base

    def load_knowledge_base(self, image_dict=None, scores_path=None, visual_attr=None):
        """Load the knowledge base.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None
        if visual_attr is not None:
            # raise NotImplementedError
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if (
            scores_path is not None
        ):  # TODO: fix the knowledge base and visual_attr is None:
            # get the image scores for each entry
            print("Loading knowledge base score from {}.".format(scores_path))
            try:
                with open(scores_path, "rb") as f:
                    image_scores = pickle.load(f)
            except:
                raise FileNotFoundError(
                    "Image scores not found, which should be a url or path to a pickle file."
                )
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base


class WikipediaKnowledgeBaseEntry:
    """Knowledge base entry for OMGM.

    Returns:
    """

    def __init__(self, entry_dict, visual_attr=None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            entry_dict: The dictionary containing the knowledge base entry.
            visual_attr: The visual attribute. Deprecated in the current version.

        Returns:
            KnowledgeBaseEntry
        """
        self.title = entry_dict["title"]
        self.url = entry_dict["url"]
        self.image_urls = entry_dict["image_urls"]
        self.image_reference_descriptions = entry_dict["image_reference_descriptions"]
        self.image_section_indices = entry_dict["image_section_indices"]
        self.section_titles = entry_dict["section_titles"]
        self.section_texts = entry_dict["section_texts"]
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr


class Retriever:
    """Retriever parent class for OMGM."""

    def __init__(self, model=None):
        """Initialize the Retriever class.

        Args:
            model: The model to use for retrieval.
        """
        self.model = model

    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        raise NotImplementedError

    def retrieve_image(self, image):
        """Retrieve the image.

        Args:
            image: The image to retrieve.
        """
        raise NotImplementedError


class ClipRetriever(Retriever):
    """Image Retriever with CLIP-based VIT."""

    def __init__(self, model="clip", device="cpu"):
        """Initialize the ClipRetriever class.

        Args:
            model: The model to use for retrieval. Should be 'clip' or 'eva-clip'.
            device: The device to use for retrieval.
        """
        super().__init__(model)
        self.model_type = model
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to("cuda").eval()
            self.processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip":
            self.model = AutoModel.from_pretrained(
                "hugging_face_cache/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ) # "BAAI/EVA-CLIP-8B"
            self.model.to("cuda").eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "hugging_face_cache/clip-vit-large-patch14"
            ) # "openai/clip-vit-large-patch14"
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "hugging_face_cache/clip-vit-large-patch14"
                )
        self.device = device
        self.model.to(device)
        self.knowledge_base = None
        

    def load_knowledge_base(
        self, knowledge_base_path, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        knowledge_base_list = self.knowledge_base.load_knowledge_base(
            image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
        )
        return knowledge_base_list
        # if scores_path is a folder, then load all the scores in the folder, otherwise, load the single score file

    def save_knowledge_base_faiss(
        self,
        knowledge_base_path,
        image_dict=None,
        scores_path=None,
        visual_attr=None,
        save_path=None,
    ):
        """Save the knowledge base with faiss index.

        Args:
            knowledge_base_path: The knowledge base to load.
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
            save_path: The path to save the faiss index.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        if scores_path[-4:] == ".pkl":
            print("Loading knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        else:
            print("Loading full knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base_full(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        self.prepare_faiss_index()
        self.save_faiss_index(save_path)


    def save_faiss_index(self, save_index_path):
        """Save the faiss index.
        
        Args:
            save_index_path: The path to save the faiss index.
        """
        if save_index_path is not None:
            write_index(self.faiss_index, save_index_path + "kb_index.faiss")
            with open(os.path.join(save_index_path, "kb_index_ids.pkl"), "wb") as f:
                pickle.dump(self.faiss_index_ids, f)


    def load_summary_faiss_index(self, load_index_path):
        """Load the summary faiss index.
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            print('Loading index...')
            self.summary_faiss_index = faiss.read_index(load_index_path)
            res = faiss.StandardGpuResources() # GPU支持
            self.summary_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.summary_faiss_index)
            print("Faiss index loaded with {} entries.".format(self.summary_faiss_index.ntotal))
        return

    def load_title_faiss_index(self, load_index_path):
        """Load the summary faiss index.
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            print('Loading index...')
            self.title_faiss_index = faiss.read_index(load_index_path)
            res = faiss.StandardGpuResources()
            self.title_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.title_faiss_index)
            print("Faiss index loaded with {} entries.".format(self.title_faiss_index.ntotal))
        return
    

    def load_image_faiss_index(self, load_index_path):
        """Load the image faiss index with pkl file
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            self.image_faiss_index = read_index(os.path.join(load_index_path, "kb_index.faiss"))
            res = faiss.StandardGpuResources()
            self.image_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.image_faiss_index)
            with open(os.path.join(load_index_path + "kb_index_ids.pkl"), "rb") as f:
                self.faiss_index_ids = pickle.load(f)
                self.faiss_index_ids_np = np.asarray(self.faiss_index_ids)

            print("Faiss index loaded with {} entries.".format(self.image_faiss_index.ntotal))
        return


    def load_section_faiss_index(self, load_index_path):
        """Load the image faiss index with pkl file
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            self.section_faiss_index = read_index(os.path.join(load_index_path, "section.index"))
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.verbose = True
            self.section_faiss_index = faiss.index_cpu_to_all_gpus(self.section_faiss_index, co=co)

            with open(os.path.join(load_index_path + "section_kb_index_ids.pkl"), "rb") as f:
                self.section_index_ids = pickle.load(f)

            print("Faiss index loaded with {} entries.".format(self.section_faiss_index.ntotal))
        return

    def prepare_faiss_index(self):
        """Prepare the faiss index from scores in the knowledge base."""
        # use the knowledge base's score element to build the index
        # get the image scores for each entry
        scores = [
            score for entry in self.knowledge_base for score in entry.score.values()
        ]
        score_ids = [
            i
            for i in range(len(self.knowledge_base))
            for j in range(len(self.knowledge_base[i].score))
        ]
        
        # import ipdb; ipdb.set_trace()
        index = faiss.IndexFlatIP(scores[0].shape[0])
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        np_scores = np.array(scores)
        np_scores = np_scores.astype(np.float32)
        faiss.normalize_L2(np_scores)
        index.add(np_scores)
        self.faiss_index = index
        self.faiss_index_ids = score_ids
        print("Faiss index built with {} entries.".format(index.ntotal))

        return

    @torch.no_grad()
    def Encoding_Query_Input(self,input_dict):
        if 'image' in input_dict and 'captions' in input_dict:
            inputs_image = self.processor(images=input_dict['image'], return_tensors="pt").pixel_values.to(self.device).half()
            inputs_caption = self.tokenizer(input_dict['captions'],  return_tensors="pt", padding=True, truncation=True,).input_ids.to('cuda')
            image_features = self.model.encode_image(inputs_image)
            captions_features= self.model.encode_text(inputs_caption)
            img_query_emb = torch.nn.functional.normalize(image_features.float())
            cap_query_emb = torch.nn.functional.normalize(captions_features.float())
            query_dict = {
                'img_query_emb':img_query_emb,
                'cap_query_emb':cap_query_emb
            }

        else:
            if self.model_type == "clip":
                inputs = (self.processor(images=input_dict['image'], return_tensors="pt").pixel_values.to(self.device).half())
                image_score = self.model.get_image_features(inputs)
            elif self.model_type == "eva-clip" or self.model_type == "eva-clip-full":
                inputs = (self.processor(images=input_dict['image'], return_tensors="pt").pixel_values.to(self.device).half())
                image_score = self.model.encode_image(inputs)
            query = image_score.float()
            query = torch.nn.functional.normalize(query)
            query_dict = {
                'img_query_emb':query
            }

        return query_dict
    

    @torch.no_grad()
    def retrieval_summary_faiss(
        self, query, top_k=100, return_entry_list=False
    ):
        """Retrieve the top K similar summary from the knowledge base using faiss with query image.：
        {'image':,'captions':}
        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        assert self.summary_faiss_index is not None
        D, I = self.summary_faiss_index.search(query.cpu().numpy(), top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            index_id = I[0][i]
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[index_id])
            else:
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[index_id].url,
                        "knowledge_base_index": int(index_id),
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[index_id],
                    }
                )
        return top_k_entries



    @torch.no_grad()
    def retrieval_image_faiss(
        self, query, top_k=20, pool_method="max", return_entry_list=False
    ):
        """Retrieve the top K similar images from the knowledge base using faiss.
        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        assert self.image_faiss_index is not None
        D, I = self.image_faiss_index.search(query.cpu().numpy(), top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            if return_entry_list:
                # map image id to entry id
                top_k_entries.append(self.knowledge_base[self.faiss_index_ids[I[0][i]]])
            else:
                # find the first knowledge base entry that contains the image
                index_id = self.faiss_index_ids[I[0][i]]
                # return the index of the first element in faiss_index_ids that is equal to index_id
                start_id = self.faiss_index_ids.index(index_id)
                offset = I[0][i] - start_id
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[self.faiss_index_ids[I[0][i]]].url,
                        "knowledge_base_index": self.faiss_index_ids[I[0][i]],
                        "image_url": self.knowledge_base[
                            self.faiss_index_ids[I[0][i]]
                        ].image_urls[offset],
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[self.faiss_index_ids[I[0][i]]],
                    }
                )
        return top_k_entries
    

    @torch.no_grad()
    def retrieval_section_faiss(
        self, query, top_k=100, pool_method="max", return_entry_list=False
    ):
        """Retrieve the top K similar images from the knowledge base using faiss.
        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        assert self.section_faiss_index is not None
        D, I = self.section_faiss_index.search(query.cpu().numpy(), top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            if return_entry_list:
                # map image id to entry id
                top_k_entries.append(self.knowledge_base[self.section_index_ids[I[0][i]]])
            else:
                # find the first knowledge base entry that contains the image
                index_id = self.section_index_ids[I[0][i]]
                # return the index of the first element in faiss_index_ids that is equal to index_id
                start_id = self.section_index_ids.index(index_id)
                offset = I[0][i] - start_id
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[self.section_index_ids[I[0][i]]].url,
                        "knowledge_base_index": self.section_index_ids[I[0][i]],
                        "section_offset":offset,
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[self.section_index_ids[I[0][i]]],
                    }
                )
        return top_k_entries



    @torch.no_grad()
    def retrieval_title_faiss(
        self, query, top_k=20, return_entry_list=False
    ):
        """Retrieve the top K similar summary from the knowledge base using faiss with query image.
        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        assert self.title_faiss_index is not None
        D, I = self.title_faiss_index.search(query.cpu().numpy(), top_k)
        top_k_entries = []
        for i in range(top_k):  # for each entity in the top k
            index_id = I[0][i]
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[index_id])
            else:
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[index_id].url,
                        "knowledge_base_index": int(index_id),
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[index_id],
                    }
                )
        return top_k_entries


    @torch.no_grad()
    def retrieval_ours(self, query, mr_tool, image_embedding, title_embedding, summa_embedding, top_k=20, return_entry_list=False):
        assert self.title_faiss_index is not None
        assert self.image_faiss_index is not None
        assert self.summary_faiss_index is not None
        query_cpu = query.cpu().numpy()
        k_search = top_k*5
        # ------------
        t_D, t_I = self.title_faiss_index.search(query_cpu, k_search)
        t_norm_score = t_D[0][:top_k]
        t_vector_id = t_I[0][:top_k]
        # ------------
        s_D, s_I = self.summary_faiss_index.search(query_cpu, k_search)
        s_norm_score = s_D[0][:top_k]
        s_vector_id = s_I[0][:top_k]
        scale_indicator = np.array(s_D[0][:top_k], dtype=np.float32)
        # -------------
        i_D, i_I = self.image_faiss_index.search(query_cpu, k_search)
        i_norm_score = i_D[0][:top_k]
        i_vector_id = i_I[0][:top_k]

        i_entity_id = self.faiss_index_ids_np[i_vector_id].tolist()
        # -------------
        raw_data_list = [{'ids':i_entity_id,'scores':i_norm_score,'vecs':image_embedding[i_vector_id]},
                         {'ids':t_vector_id,'scores':t_norm_score,'vecs':title_embedding[t_vector_id]},
                         {'ids':s_vector_id,'scores':s_norm_score,'vecs':summa_embedding[s_vector_id]}]
        final_ranking = mr_tool.run(raw_data_list)
        top_final = final_ranking[:top_k]
        raw_scores = np.fromiter((float(sc) for _, sc in top_final), dtype=np.float32, count=len(top_final))

        mapped_scores = self.quantile_match_np(raw_scores, scale_indicator)

        if return_entry_list:
            return [self.knowledge_base[int(idx)] for idx, _ in top_final]

        top_k_entries = []
        for i, (idx, _) in enumerate(top_final):
            idx = int(idx)
            entry = self.knowledge_base[idx]
            top_k_entries.append({
                "url": entry.url,
                "knowledge_base_index": idx,
                "similarity": float(mapped_scores[i]),
                "kb_entry": entry,
            })
        return top_k_entries
    
 
    @staticmethod
    def quantile_match_np(x, ref, eps=1e-12):
        x = np.asarray(x, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)

        K = len(x)
        order = np.argsort(x)
        ranks = np.empty(K, dtype=np.int64)
        ranks[order] = np.arange(K)

        q = ranks / max(K - 1, 1)  # [0,1]
        ref_sorted = np.sort(ref)

        pos = q * (K - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        w = (pos - lo).astype(np.float32)

        out = (1 - w) * ref_sorted[lo] + w * ref_sorted[hi]
        return out



class MBEIRClipRetriever(Retriever):
    """Image Retriever with CLIP-based VIT."""

    def __init__(self, model="clip", device="cpu"):
        """Initialize the ClipRetriever class.

        Args:
            model: The model to use for retrieval. Should be 'clip' or 'eva-clip'.
            device: The device to use for retrieval.
        """
        super().__init__(model)
        self.model_type = model
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to("cuda").eval()
            self.processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip":
            self.model = AutoModel.from_pretrained(
                "hugging_face_cache/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ) # "BAAI/EVA-CLIP-8B"
            self.model.to("cuda").eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "hugging_face_cache/clip-vit-large-patch14"
            ) # "openai/clip-vit-large-patch14"
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "hugging_face_cache/clip-vit-large-patch14"
                )
        self.device = device
        self.model.to(device)
        self.knowledge_base = None
        

    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
            The knowledge path is a jsonl file
        """
        self.knowledge_base = [json.loads(line) for line in open(knowledge_base_path, encoding="utf-8")]
        return None
        # if scores_path is a folder, then load all the scores in the folder, otherwise, load the single score file


    def load_text_faiss_index(self, load_index_path):
        """Load the summary faiss index.
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            print('Loading index...')
            self.text_faiss_index = faiss.read_index(load_index_path)
            res = faiss.StandardGpuResources()
            self.text_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.text_faiss_index)
            print("Faiss index loaded with {} entries.".format(self.text_faiss_index.ntotal))
        return
    

    def load_image_faiss_index(self, load_index_path):
        """Load the summary faiss index.
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            print('Loading index...')
            self.image_faiss_index = faiss.read_index(load_index_path)
            res = faiss.StandardGpuResources()
            self.image_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.image_faiss_index)
            print("Faiss index loaded with {} entries.".format(self.image_faiss_index.ntotal))
        return
    

    @torch.no_grad()
    def Encoding_Query_Input(self,input_dict):
        if 'image' in input_dict and 'captions' in input_dict:
            inputs_image = self.processor(images=input_dict['image'], return_tensors="pt").pixel_values.to(self.device).half()
            inputs_caption = self.tokenizer(input_dict['captions'],  return_tensors="pt", padding=True, truncation=True,).input_ids.to('cuda')
            image_features = self.model.encode_image(inputs_image)
            captions_features= self.model.encode_text(inputs_caption)
            img_query_emb = torch.nn.functional.normalize(image_features.float())
            cap_query_emb = torch.nn.functional.normalize(captions_features.float())
            query_dict = {
                'img_query_emb':img_query_emb,
                'cap_query_emb':cap_query_emb
            }
        elif 'image' not in input_dict and 'captions' in input_dict:
            inputs_caption = self.tokenizer(input_dict['captions'],  return_tensors="pt", padding=True, truncation=True,).input_ids.to('cuda')
            captions_features= self.model.encode_text(inputs_caption)
            cap_query_emb = torch.nn.functional.normalize(captions_features.float())
            query_dict = {
                'cap_query_emb':cap_query_emb
            }

        else:
            if self.model_type == "clip":
                inputs = (self.processor(images=input_dict['image'], return_tensors="pt").pixel_values.to(self.device).half())
                image_score = self.model.get_image_features(inputs)
            elif self.model_type == "eva-clip" or self.model_type == "eva-clip-full":
                inputs = (self.processor(images=input_dict['image'], return_tensors="pt").pixel_values.to(self.device).half())
                image_score = self.model.encode_image(inputs)
            query = image_score.float()
            query = torch.nn.functional.normalize(query)
            query_dict = {
                'img_query_emb':query
            }

        return query_dict
    

    @torch.no_grad()
    def retrieval_text_faiss(self, query, top_k=20, return_entry_list=False):
        """Retrieve the top K similar summary from the knowledge base using faiss with query image.
        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        assert self.text_faiss_index is not None
        D, I = self.text_faiss_index.search(query.cpu().numpy(), top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            index_id = I[0][i]
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[index_id])
            else:
                top_k_entries.append(
                    {
                        "id": self.knowledge_base[index_id]['did'],
                        "knowledge_base_index": int(index_id),
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[index_id],
                    }
                )
        return top_k_entries
    

    @torch.no_grad()
    def retrieval_image_faiss(self, query, top_k=20, return_entry_list=False):
        """Retrieve the top K similar summary from the knowledge base using faiss with query image.
        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        assert self.image_faiss_index is not None
        D, I = self.image_faiss_index.search(query.cpu().numpy(), top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            index_id = I[0][i]
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[index_id])
            else:
                top_k_entries.append(
                    {
                        "id": self.knowledge_base[index_id]['did'],
                        "knowledge_base_index": int(index_id),
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[index_id],
                    }
                )
        return top_k_entries


    @torch.no_grad()
    def retrieval_ours(self, query, mr_tool, text_embedding, image_embedding, top_k=20, return_entry_list=False):
        assert self.text_faiss_index is not None
        assert self.image_faiss_index is not None
        query_cpu = query.cpu().numpy()
        # ------------
        t_D, t_I = self.text_faiss_index.search(query_cpu, top_k)
        t_norm_score = t_D[0][:top_k]
        t_vector_id = t_I[0][:top_k]
        t_entity_id = [self.knowledge_base[index_id]['did'] for index_id in t_vector_id]
        # ------------
        i_D, i_I = self.image_faiss_index.search(query_cpu, top_k)
        i_norm_score = i_D[0][:top_k]
        i_vector_id = i_I[0][:top_k]
        i_entity_id = [self.knowledge_base[index_id]['did'] for index_id in i_vector_id]
        # ------------
        # -------------
        raw_data_list = [{'ids':t_vector_id,'scores':t_norm_score,'vecs':text_embedding[t_vector_id]},
                         {'ids':i_vector_id,'scores':i_norm_score,'vecs':image_embedding[i_vector_id]},]
        final_ranking = mr_tool.run(raw_data_list)

        top_final = final_ranking[:top_k]
        raw_scores = np.fromiter((float(sc) for _, sc in top_final), dtype=np.float32, count=len(top_final))

        top_k_entries = []
        for i, (idx, _) in enumerate(top_final):
            idx = int(idx)
            entry = self.knowledge_base[idx]
            top_k_entries.append({
                "id": self.knowledge_base[idx]['did'],
                "knowledge_base_index": idx,
                "similarity": float(raw_scores[i]),
                "kb_entry": entry,

            })
        additional = [{'t_ids':list(t_entity_id),'t_scores':list(t_norm_score), 't_embedding':list(t_vector_id)},
                         {'s_ids':list(i_entity_id),'s_scores':list(i_norm_score), 's_embedding':list(i_vector_id)}]
        return top_k_entries, additional

    @staticmethod
    def quantile_match_np(x, ref, eps=1e-12):
        x = np.asarray(x, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)

        K = len(x)
        order = np.argsort(x)
        ranks = np.empty(K, dtype=np.int64)
        ranks[order] = np.arange(K)

        q = ranks / max(K - 1, 1)  # [0,1]
        ref_sorted = np.sort(ref)

        pos = q * (K - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        w = (pos - lo).astype(np.float32)

        out = (1 - w) * ref_sorted[lo] + w * ref_sorted[hi]
        return out
