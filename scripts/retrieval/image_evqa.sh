export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TF_ENABLE_ONEDNN_OPTS=0

python -m test.step1_coarse_retrieval_kbvqa \
    --sample_file ../vqa_data/evqa_data/vqa_test.csv\
    --knowledge_base ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json\
    --faiss_index ../vqa_data/evqa_image_index/\
    --save_result_path results/step1_results/E-VQA/image_retrieval.json\
    --retriever_vit eva-clip \
    --top_ks 1,5,10,20 \
    --retrieval_top_k 20\
    --index_modal image\