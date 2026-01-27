export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/user/seraveea/research/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"


python -m test.step2_fine_rerank \
    --sample_file ../vqa_data/infoseek_data/infoseek_test_filtered.csv\
    --step1_result results/step1_results/infoseek/image_retrieval.json \
    --reranker_ckpt_path model_weights/model_23912.pth \
    --knowledge_base_path ../vqa_data/infoseek_kb/wiki_100_dict_v4.json\
    --top_ks 1,5,10,20 \
    --step1_alpha  0.9 \
    --wiki_img_csv_dir ../vqa_data/KB_image/output/ \
    --wiki_img_path_prefix ../vqa_data/KB_image/ \
    --save_result_path results/step2_results/infoseek/image_rerank.json \
