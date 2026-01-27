import os
import json
import argparse
from tqdm import tqdm

from utils import evaluate_example, evaluate, load_csv_data, get_test_question


def evaluate_and_dump(
    image_dataset_name: str,
    result_list: list,
    result_json_path: str,
    out_dir: str,
) -> str:
    """
    Evaluate result_list for a given dataset and dump metrics to a json file.

    Output file name is derived from result_json_path:
        xxx.json -> xxx_eval.json
    """
    assert image_dataset_name in {"evqa", "infoseek"}, f"Unsupported dataset: {image_dataset_name}"
    assert isinstance(result_list, list) and len(result_list) > 0

    base = "/data/user/seraveea/research/vqa_data"

    # ---------------- evaluation ----------------
    if image_dataset_name == "infoseek":
        gt_jsonl = os.path.join(base, "infoseek_data/infoseek_val.jsonl")
        pred_jsonl = os.path.join(base, "infoseek_data/infoseek_val.jsonl")

        result = evaluate(result_json_path, gt_jsonl, pred_jsonl, True)

        metrics = {
            "dataset": "infoseek",
            "final_score": result.get("final_score"),
            "unseen_question_score": result.get("unseen_question_score", {}).get("score"),
            "unseen_entity_score": result.get("unseen_entity_score", {}).get("score"),
            "detail": {
                "unseen_question_score_time": result.get("unseen_question_score", {}).get("score_time"),
                "unseen_question_score_num": result.get("unseen_question_score", {}).get("score_num"),
                "unseen_question_score_string": result.get("unseen_question_score", {}).get("score_string"),
                "unseen_entity_score_time": result.get("unseen_entity_score", {}).get("score_time"),
                "unseen_entity_score_num": result.get("unseen_entity_score", {}).get("score_num"),
                "unseen_entity_score_string": result.get("unseen_entity_score", {}).get("score_string"),
            },
        }

    else:
        test_file = os.path.join(base, "evqa_data/vqa_test.csv")
        test_list, test_header = load_csv_data(test_file)

        total_score = 0.0
        n = len(test_list)

        for it, _ in tqdm(enumerate(test_list), total=n):
            sample = get_test_question(it, test_list, test_header)

            data_id = f"E-VQA_{it}"
            if result_list[it].get("data_id") != data_id:
                raise ValueError(
                    f"Data ID mismatch at idx={it}: "
                    f"{result_list[it].get('data_id')} vs {data_id}"
                )

            total_score += evaluate_example(
                sample["question"],
                reference_list=sample["answer"].split("|"),
                candidate=result_list[it]["prediction"],
                question_type=sample["question_type"],
            )

        metrics = {
            "dataset": "evqa",
            "avg_score": total_score / max(1, n),
            "sum_score": total_score,
            "num_samples": n,
        }

    # ---------------- output name ----------------
    basename = os.path.basename(result_json_path)
    stem = os.path.splitext(basename)[0]      # remove .json
    out_name = f"{image_dataset_name}_{stem}_eval.json"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser("VQA Evaluation Script")
    parser.add_argument("--dataset", default="infoseek", choices=["evqa", "infoseek"])
    parser.add_argument("--result_json", default="results/step3_results_w_rerank/infoseek/rerank_llava_summary_generation.json", type=str)
    parser.add_argument("--out_dir", default="results/eval_scores", type=str)

    args = parser.parse_args()

    with open(args.result_json, "r", encoding="utf-8") as f:
        result_list = json.load(f)

    out_path = evaluate_and_dump(
        image_dataset_name=args.dataset,
        result_list=result_list,
        result_json_path=args.result_json,
        out_dir=args.out_dir,
    )

    print(f"Evaluation finished")
    print(f"    Dataset : {args.dataset}")
    print(f"    Output  : {out_path}")


if __name__ == "__main__":
    main()