# SCAR: Structure-aware Cross-view Alignment for multi-modal Retrieval

Official implementation of **SCAR** (Structure-aware Cross-view Alignment for multi-modal Retrieval), a novel multi-view retrieval method for Knowledge-Based Visual Question Answering (KB-VQA) tasks.


## 📋 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)



## Introduction

**SCAR** is a multi-view retrieval method designed for encyclopedic visual question answering. Given a query image and question, SCAR retrieves relevant knowledge from large-scale Wikipedia knowledge bases by:

- **Similarity Propagation via Entity kNN
Graphs**: mprove single view performance via similarity propagation (title, sections, images)
- **Cross-View Redundancy Regulation**: Consolidate complementary signals via suppressing redundancy.


The complete system follows a three-stage pipeline (adapted from [OMGM](https://github.com/ChaoLinAViy/OMGM) and [EchoSight](https://github.com/Go2Heart/EchoSight)):

1. **Stage 1: Coarse Retrieval** - **SCAR method** rapidly retrieves candidate documents from the knowledge base
2. **Stage 2: Fine-grained Reranking** - Reranks sections within candidate documents for more precise context
3. **Stage 3: Answer Generation** - Generates final answers using Vision-Language Models (VLMs)



## Key Features

✨ **SCAR: Our Core Method**
- Multi-view retrieval combining textual (title, summary) and visual (images) signals
- Structure-aware cross-view alignment mechanism
- Graph-enhanced retrieval with adaptive weight propagation
- Significantly outperforms single-view baselines

🔍 **Three-Stage Pipeline**
- Stage 1: SCAR-based coarse retrieval
- Stage 2: Section-level fine-grained reranking
- Stage 3: Context-aware answer generation

🤖 **Multiple VLM Support**
- LLaVA 1.5
- Qwen3-VL
- InternVL 2.5 / 3.5

## Installation

### Requirements

- Python 3.10+
- CUDA 11.0+ (for GPU acceleration)


### Dependencies Installation

```bash
# Clone the repository
git clone https://github.com/[will release after double-blind review].git
cd SCAR_official

# Create virtual environment (recommended)
conda create -n scar python=3.10
conda activate scar

# Install dependencies
pip install -r requirements.txt
```
> **Important**: We have modified the LLaVA codebase to ensure compatibility with higher versions of Transformers. Please use the `llava/` and `lavis/` folders included in this repository instead of the original implementations.

> **Compatibility Note**: The pre-trained reranker models are incompatible with newer Transformers versions required by modern MLLMs (e.g., Qwen3-VL). If you need to perform OMGM-style reranking, please create a separate virtual environment using `rerank_requirements.txt`:

```bash
# Create separate environment for reranking
conda create -n scar_rerank python=3.10
conda activate scar_rerank
pip install -r rerank_requirements.txt
```


### Model Checkpoints

Download the following pre-trained models:

1. **Retrieval Models**
- EVA-CLIP: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EVA--CLIP--8B-blue)](https://huggingface.co/BAAI/EVA-CLIP-8B)


2. **Generation Models**
- LLaVA-1.5: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LLaVA--1.5--7B-blue)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Qwen3-VL: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen3--VL--8B-blue)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- InternVL3.5: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-InternVL3.5--8B-blue)](https://huggingface.co/OpenGVLab/InternVL3_5-8B)

3. **Reranking Models**
- BGE Reranker: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BGE--Reranker-blue)](https://huggingface.co/BAAI/bge-reranker-v2-m3)

- QFormer IT2IT Reranker (from [OMGM](https://github.com/ChaoLinAViy/OMGM))
> **Note**: After downloading the models, update the model paths in the configuration files and scripts to match your local storage locations. Or point to a huggingface cache directory.



## Data Preparation

### 1. Download Datasets

We highly recommend referencing the [OMGM](https://github.com/ChaoLinAViy/OMGM), [EchoSight](https://github.com/Go2Heart/EchoSight) and [ReflectiVA](https://github.com/aimagelab/ReflectiVA) repositories for detailed instructions on downloading and preparing the E-VQA and InfoSeek datasets.

### 2. Build FAISS Indices

We use pre-built FAISS indices from both [EchoSight](https://github.com/Go2Heart/EchoSight) and [OMGM](https://github.com/ChaoLinAViy/OMGM).
For Title based index, we will release after double-blind review.
Or you can build them from scratch using our code:

```bash
cd faiss_build

# Build image index
python faiss_build.py \
    --mode image \
    --knowledge_base ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json \
    --output_index ../vqa_data/evqa_image_index/

# Build text index
python faiss_build.py \
    --mode title \
    --knowledge_base ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json \
    --output_index ../vqa_data/evqa_title.index

# Build summary index
python faiss_build.py \
    --mode summary \
    --knowledge_base ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json \
    --output_index ../vqa_data/evqa_summary_llama3.index
```

## Usage

### Complete Pipeline

#### 1. Stage 1: SCAR Coarse Retrieval 

Take E-VQA as example.

SCAR performs structure-aware multi-view retrieval by combining title, summary, and image-based signals.

```bash
bash scripts/retrieval/ours_evqa.sh
```

#### 2. Stage 2: Fine-grained Reranking
We use the reranker models and pre-trained weights provided by OMGM. Please activate the reranking environment and download the reranker weights from the links provided in [OMGM repository](https://github.com/ChaoLinAViy/OMGM/tree/master).

```bash
# Activate reranking environment
conda activate scar_rerank

# Run reranking
```bash
bash scripts/fine_reranking/evqa/evqa_ours.sh
```

#### 3. Stage 3: Answer Generation

```bash
bash scripts/answer_generation/evqa_answer_ours.sh
```

#### 4. Evaluate Results

```bash
python -m test.step4_generation_evaluation \
    --result_file results/step3_results/E-VQA/generation.json \
    --dataset evqa
```

## Project Structure

```
SCAR_official/
├── data_utils.py              # Data processing utilities
├── dataset/                   # Dataset loading modules
│   ├── collator.py
│   ├── dataset.py
│   └── dataset_utils.py
├── faiss_build/               # FAISS index construction
│   ├── faiss_build.py
│   ├── mb_faiss_build.py
│   └── summary_generation.py
├── lavis/                     # LAVIS framework integration
│   ├── models/               # BLIP, BLIP2 models
│   ├── processors/           # Image/text processors
│   └── configs/              # Model configurations
├── llava/                     # LLaVA model integration
│   ├── model/
│   └── eval/
├── model/                     # Core model implementations
│   ├── retriever.py          # Retriever module (SCAR implementation)
│   ├── router.py             # Multi-view routing (SCAR core)
│   └── answer_generator.py   # Answer generator
├── scripts/                   # Execution scripts
│   ├── retrieval/            # Retrieval scripts
│   ├── fine_reranking/       # Reranking scripts
│   └── answer_generation/    # Generation scripts
├── test/                      # Testing and evaluation
│   ├── step1_coarse_retrieval_kbvqa.py  # SCAR retrieval
│   ├── step2_fine_rerank.py
│   ├── step3_answer_generating.py
│   └── step4_generation_evaluation.py
└── utils/                     # Utility functions
    ├── utils.py
    ├── evqa_evaluation_utils.py
    └── infoseek_evaluation_utils.py
```

## Acknowledgments

This project builds upon the following open-source projects:
- [OMGM](https://github.com/ChaoLinAViy/OMGM) - Three-stage KB-VQA framework
- [LAVIS](https://github.com/salesforce/LAVIS) - Vision-Language models
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Large Language and Vision Assistant
- [EchoSight](https://github.com/Go2Heart/EchoSight) - Image-view indexing and retrieval
- [ReflectiVA](https://github.com/aimagelab/ReflectiVA) - Evaluation codebase
- [Encyclopedic-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa) - Dataset
- [InfoSeek](https://github.com/open-vision-language/infoseek) - Dataset
