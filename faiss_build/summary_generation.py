import json
import torch
import argparse
from tqdm import tqdm
from transformers import pipeline
import faiss
import torch
import numpy as np
from transformers import AutoModel, CLIPTokenizer
from model.retriever import WikipediaKnowledgeBaseEntry
import pickle


def reconstruct_wiki_page(knowledge_entry, section_index=-1):
    """Reconstruct the wiki sections from the knowledge entry class."""
    title = knowledge_entry.title
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if it == int(section_index):
            evidence_section = (
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry.section_texts[it]
            )
        elif (
            "external links" in section_title.lower()
            or "references" in section_title.lower()
        ):
            continue
        else:
            sections.append(
                "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry.section_texts[it]
            )
    if section_index != -1:
        return evidence_section, sections
    return "\n\n".join(sections)

def pipeline_instance(args):
    model_dir = "hugging_face_cache/llama-3-70B-Instruct"
    gen_pipeline = pipeline(
        "text-generation",
        model=model_dir,
        device_map=args.device,
        torch_dtype=torch.float16 
    )
    return gen_pipeline

def agent_reply(llm, wiki_content):
    message = [
        {"role":"System:","content":"""
         You are a Wiki Summary Generator Assistant. Following is some information about you: 
         ## Profile - name: Wiki Summary Generator Assistant 
         - language: English 
         - description: The Wiki Summary Generator Assistant is designed to create concise and informative summaries based on provided Wikipedia content. 
         It extracts key aspects of the entity mentioned in the Wiki article, covering various dimensions such as history, characteristics, significance, appearance and impact. 
         ## Workflows 1. Input the provided Wikipedia content into the system. 
         2. Identify the main sections and key information related to the entity. 
         3. Synthesize this information into a well-structured summary. 
         4. Review and refine the summary for clarity, coherence, and completeness before finalizing. 
         ## Rules 1. Focus on summarizing key details across multiple aspects (e.g., appearance, features, impact) of the entity. 
         2. Ensure the summary is concise, clear, and free of irrelevant details. 
         3. Retain the original meaning and context of the Wiki content while rephrasing it into a summary."""},
        {"role":"User:","content":f"""Following is the input Wikipedia content:{wiki_content}
         """}
    ]
    eos_pair = [llm.tokenizer.eos_token_id, llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # llama
    prompt = llm.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    output = llm(prompt, max_new_tokens=2048, 
                            eos_token_id=eos_pair, 
                            do_sample=False, 
                            # temperature=temperature, 
                            # top_p=0.9, 
                            pad_token_id=llm.tokenizer.eos_token_id)
    return output[0]["generated_text"][len(prompt):]

with open("../vqa_data/evqa_kb/encyclopedic_kb_wiki.json", "r") as f:
    kb_entries = json.load(f)

kb_base = [WikipediaKnowledgeBaseEntry(entry) for _, entry in kb_entries.items()]
print(f"[INFO] Total KB entries: {len(kb_base)}")

for entry in tqdm(kb_base, desc="Building FAISS index"):
    content = reconstruct_wiki_page(entry)
    print(content)
    break
