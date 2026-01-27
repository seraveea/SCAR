export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

python -m test.step3_answer_generating \
    --test_file ../vqa_data/evqa_data/vqa_test.csv\
    --retrieval_results_file results/step2_results/E-VQA/ours_rerank.json \
    --step2_beta 0.2 \
    --answer_generator qwen3-vl \
    --knowledge_base_path ../vqa_data/evqa_kb/encyclopedic_kb_wiki.json\
    --output_file results/step3_results/E-VQA/qwen3-vl_ours_generation.json \
    --llm_checkpoint None \
