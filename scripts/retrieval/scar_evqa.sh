

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TF_ENABLE_ONEDNN_OPTS=0

python -m test.step1_coarse_retrieval_kbvqa \
    --sample_file ../vqa_data/evqa_data/vqa_test.csv\
    --retrieval_top_k 20\
    --knowledge_base ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json\
    --faiss_index ../vqa_data/evqa_title.index\
    --save_result_path results/step1_results/E-VQA/ours_retrieval.json\
    --retriever_vit eva-clip\
    --top_ks 1,5,10,20 \
    --index_modal ours\