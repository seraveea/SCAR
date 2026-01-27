export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TF_ENABLE_ONEDNN_OPTS=0

python faiss_build/summary_generation.py