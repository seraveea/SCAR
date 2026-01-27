export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export HF_HUB_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub"


# modal: text, image, ours
# datasets: WebQA,OVEN
python -m test.step1_MBEIR_retrieval \
    --dataset WebQA\
    --index_modal ours\
    --retriever_vit eva-clip\
    --top_ks 1,5,10,20 \
    --retrieval_top_k 20\
    --save_result_path results/mbeir_results/webqa_ours.json
