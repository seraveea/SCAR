export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/user/seraveea/research/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export HF_HUB_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub"


# llava1_5, internvl3_5, qwen3-vl
python -m test.step3_answer_generating \
    --test_file ../vqa_data/evqa_data/vqa_test.csv\
    --retrieval_results_file results/step2_results/E-VQA/image_rerank.json \
    --step2_beta 0.2 \
    --answer_generator qwen3-vl \
    --knowledge_base_path ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json\
    --output_file results/step3_results/E-VQA/qwen3-vl_image_generation.json \
    --llm_checkpoint None \
