export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TF_ENABLE_ONEDNN_OPTS=0

python -m test.step1_coarse_retrieval_kbvqa \
    --sample_file ../vqa_data/infoseek_data/infoseek_test_filtered.csv\
    --knowledge_base ../vqa_data/infoseek_kb/wiki_100_dict_v4.json\
    --faiss_index "../vqa_data/infoseek_image_index/"\
    --save_result_path None\
    --retriever_vit eva-clip\
    --top_ks 1,5,10,20\
    --index_modal image\


# --save_result_path results/step1_results/infoseek/image_retrieval.json\