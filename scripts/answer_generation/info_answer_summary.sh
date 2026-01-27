export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/user/seraveea/research/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"


# llava1_5, internvl3_5, qwen3-vl
python -m test.step3_answer_generating \
    --test_file ../vqa_data/infoseek_data/infoseek_test_filtered.csv\
    --retrieval_results_file results/step1_results/infoseek/summary_retrieval.json \
    --step2_beta 0.2 \
    --answer_generator internvl3_5 \
    --knowledge_base_path ../vqa_data/infoseek_kb/wiki_100_dict_v4.json\
    --output_file results/step3_results/infoseek/internvl_summary_generation.json \
    --llm_checkpoint None \