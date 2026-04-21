from __future__ import annotations

from typing import Any


PAPER_TABLES: dict[str, Any] = {
    "mhalubench": {
        "title": "Table 1",
        "rows": {
            "Gemini with Self-Check Claim": {
                "level": "claim",
                "accuracy": 74.74,
                "precision": 75.80,
                "recall": 75.68,
                "macro_f1": 74.74,
            },
            "Gemini with Self-Check Segment": {
                "level": "segment",
                "accuracy": 75.11,
                "precision": 73.89,
                "recall": 77.44,
                "macro_f1": 73.85,
            },
            "Gemini with UNIHD Claim": {
                "level": "claim",
                "accuracy": 77.41,
                "precision": 77.76,
                "recall": 77.99,
                "macro_f1": 77.39,
            },
            "Gemini with UNIHD Segment": {
                "level": "segment",
                "accuracy": 78.68,
                "precision": 75.97,
                "recall": 78.64,
                "macro_f1": 76.74,
            },
            "GPT-4V with Self-Check Claim": {
                "level": "claim",
                "accuracy": 79.25,
                "precision": 79.02,
                "recall": 79.16,
                "macro_f1": 79.08,
            },
            "GPT-4V with Self-Check Segment": {
                "level": "segment",
                "accuracy": 80.80,
                "precision": 77.80,
                "recall": 78.30,
                "macro_f1": 78.04,
            },
            "GPT-4V with UNIHD Claim": {
                "level": "claim",
                "accuracy": 81.91,
                "precision": 81.81,
                "recall": 81.52,
                "macro_f1": 81.63,
            },
            "GPT-4V with UNIHD Segment": {
                "level": "segment",
                "accuracy": 84.60,
                "precision": 82.77,
                "recall": 80.89,
                "macro_f1": 81.71,
            },
            "Our Detection Model Claim": {
                "level": "claim",
                "accuracy": 85.60,
                "precision": 85.46,
                "recall": 85.79,
                "macro_f1": 85.52,
            },
            "Our Detection Model Segment": {
                "level": "segment",
                "accuracy": 86.94,
                "precision": 87.73,
                "recall": 82.88,
                "macro_f1": 85.23,
            },
        },
        "comparison_row": {
            "claim": "Our Detection Model Claim",
            "segment": "Our Detection Model Segment",
        },
    },
    "mfhallubench": {
        "title": "Table 2",
        "rows": {
            "GPT-4V 2shot": {
                "binary_precision": 59.7,
                "binary_recall": 98.7,
                "binary_accuracy": 63.3,
                "binary_f1": 74.4,
                "multi_accuracy": 40.6,
            },
            "LLaVA-1.6-34B 2shot": {
                "binary_precision": 55.5,
                "binary_recall": 100.0,
                "binary_accuracy": 56.7,
                "binary_f1": 71.4,
                "multi_accuracy": 36.7,
            },
            "Our detection model": {
                "binary_precision": 87.8,
                "binary_recall": 88.8,
                "binary_accuracy": 87.3,
                "binary_f1": 88.2,
                "multi_accuracy": 74.3,
            },
        },
        "comparison_row": "Our detection model",
    },
    "mitigation": {
        "title": "Table 3",
        "rows": {
            "LRV": {"chairs": 32.3, "chairi": 22.3},
            "POVID": {
                "chairs": 48.1,
                "chairi": 24.4,
                "amber_chair": 7.3,
                "amber_cover": 49.5,
                "amber_hal": 31.1,
                "amber_cog": 3.7,
                "mmhal_overall": 2.08,
                "mmhal_resp": 0.56,
                "pope_adv_f1": 81.6,
            },
            "InstructBLIP": {
                "chairs": 25.9,
                "chairi": 14.3,
                "amber_chair": 8.8,
                "amber_cover": 52.2,
                "amber_hal": 38.2,
                "amber_cog": 4.4,
                "mmhal_overall": 2.14,
                "mmhal_resp": 0.58,
                "pope_adv_f1": 78.4,
            },
            "Qwen-VL-Chat": {
                "chairs": 36.0,
                "chairi": 21.3,
                "amber_chair": 6.6,
                "amber_cover": 53.2,
                "amber_hal": 31.0,
                "amber_cog": 2.9,
                "mmhal_overall": 2.89,
                "mmhal_resp": 0.43,
                "llava_bench_overall": 79.8,
                "pope_adv_f1": 82.8,
            },
            "LLaVA-1.5": {
                "chairs": 46.3,
                "chairi": 22.6,
                "amber_chair": 7.8,
                "amber_cover": 51.0,
                "amber_hal": 36.4,
                "amber_cog": 4.2,
                "mmhal_overall": 2.42,
                "llava_bench_overall": 72.5,
                "pope_adv_f1": 84.5,
            },
            "LLaVA-RLHF": {
                "chairs": 38.1,
                "chairi": 18.9,
                "amber_chair": 7.7,
                "amber_cover": 52.1,
                "amber_hal": 39.0,
                "amber_cog": 4.4,
                "mmhal_overall": 2.53,
                "mmhal_resp": 0.57,
                "llava_bench_overall": 76.9,
                "pope_adv_f1": 80.5,
            },
            "RLHF-V": {
                "chairs": 12.2,
                "chairi": 7.5,
                "amber_chair": 6.3,
                "amber_cover": 46.1,
                "amber_hal": 25.1,
                "amber_cog": 2.1,
                "mmhal_overall": 2.81,
                "mmhal_resp": 0.49,
            },
            "GPT-4V": {
                "chairs": 13.6,
                "chairi": 7.3,
                "amber_chair": 4.6,
                "amber_cover": 67.1,
                "amber_hal": 30.7,
                "amber_cog": 2.6,
                "mmhal_overall": 3.49,
                "mmhal_resp": 0.28,
            },
            "Silkie": {
                "chairs": 25.3,
                "chairi": 13.9,
                "amber_chair": 5.4,
                "amber_cover": 55.8,
                "amber_hal": 29.0,
                "amber_cog": 2.0,
                "mmhal_overall": 3.01,
                "mmhal_resp": 0.41,
                "llava_bench_overall": 84.9,
                "pope_adv_f1": 82.1,
            },
            "DPO w/ Qwen-VL": {
                "chairs": 14.3,
                "chairi": 8.0,
                "amber_chair": 3.8,
                "amber_cover": 53.2,
                "amber_hal": 19.7,
                "amber_cog": 1.8,
                "mmhal_overall": 2.98,
                "mmhal_resp": 0.38,
                "llava_bench_overall": 82.0,
                "pope_adv_f1": 82.6,
            },
            "DPO w/ LLaVA-1.5": {
                "chairs": 6.7,
                "chairi": 3.6,
                "amber_chair": 2.8,
                "amber_cover": 47.8,
                "amber_hal": 15.5,
                "amber_cog": 1.6,
                "mmhal_overall": 2.58,
                "mmhal_resp": 0.50,
                "llava_bench_overall": 79.3,
                "pope_adv_f1": 84.5,
            },
            "HSA-DPO w/ Qwen-VL": {
                "chairs": 11.0,
                "chairi": 5.5,
                "amber_chair": 3.7,
                "amber_cover": 52.4,
                "amber_hal": 19.0,
                "amber_cog": 1.6,
                "mmhal_overall": 3.07,
                "mmhal_resp": 0.34,
                "llava_bench_overall": 82.4,
                "pope_adv_f1": 82.9,
            },
            "HSA-DPO w/ LLaVA-1.5": {
                "chairs": 5.3,
                "chairi": 3.2,
                "amber_chair": 2.1,
                "amber_cover": 47.3,
                "amber_hal": 13.4,
                "amber_cog": 1.2,
                "mmhal_overall": 2.61,
                "mmhal_resp": 0.48,
                "llava_bench_overall": 80.5,
                "pope_adv_f1": 84.9,
            },
        },
        "base_row": "LLaVA-1.5",
        "comparison_row": "HSA-DPO w/ LLaVA-1.5",
    },
    "ablation": {
        "title": "Table 4",
        "rows": {
            "Ours": {
                "hss": 0.60,
                "chairs": 5.3,
                "chairi": 3.2,
                "amber_chair": 2.1,
                "amber_cover": 47.3,
                "amber_hal": 13.4,
                "amber_cog": 1.2,
            },
            "w/o Detection": {
                "hss": 0.79,
                "chairs": 42.1,
                "chairi": 20.3,
                "amber_chair": 7.6,
                "amber_cover": 52.1,
                "amber_hal": 32.4,
                "amber_cog": 4.0,
            },
            "w/o HSA": {
                "hss": 0.65,
                "chairs": 6.7,
                "chairi": 3.6,
                "amber_chair": 2.8,
                "amber_cover": 47.8,
                "amber_hal": 15.5,
                "amber_cog": 1.6,
            },
            "w/o FineGrained": {
                "hss": 0.67,
                "chairs": 17.0,
                "chairi": 8.9,
                "amber_chair": 5.0,
                "amber_cover": 53.6,
                "amber_hal": 27.3,
                "amber_cog": 1.7,
            },
        },
        "comparison_row": "Ours",
    },
    "efficiency": {
        "title": "Table 5",
        "rows": {
            "Human": {"train_time": 0.0, "efficiency_seconds": 20.0, "cost_usd": 4800},
            "GPT-4V": {"train_time": 0.0, "efficiency_seconds": 8.0, "cost_usd": 1600},
            "Our pipeline": {"train_time": 5.7, "efficiency_seconds": 5.0, "cost_usd": 600},
        },
    },
}


PAPER_METRIC_DIRECTIONS: dict[str, str] = {
    "accuracy": "higher_better",
    "claim_accuracy": "higher_better",
    "claim_precision": "higher_better",
    "claim_recall": "higher_better",
    "claim_macro_f1": "higher_better",
    "precision": "higher_better",
    "recall": "higher_better",
    "f1": "higher_better",
    "macro_f1": "higher_better",
    "segment_accuracy": "higher_better",
    "segment_precision": "higher_better",
    "segment_recall": "higher_better",
    "segment_macro_f1": "higher_better",
    "binary_precision": "higher_better",
    "binary_recall": "higher_better",
    "binary_accuracy": "higher_better",
    "binary_f1": "higher_better",
    "multi_accuracy": "higher_better",
    "chairs": "lower_better",
    "chairi": "lower_better",
    "amber_chair": "lower_better",
    "amber_cover": "higher_better",
    "amber_hal": "lower_better",
    "amber_cog": "lower_better",
    "mmhal_overall": "higher_better",
    "mmhal_resp": "lower_better",
    "llava_bench_overall": "higher_better",
    "hss": "lower_better",
    "avg_hss": "lower_better",
    "sum_hss": "lower_better",
    "avg_response_length": "lower_better",
}


def mitigation_reference_row() -> str:
    return PAPER_TABLES["mitigation"]["comparison_row"]


def mitigation_base_row() -> str:
    return PAPER_TABLES["mitigation"]["base_row"]


def paper_reference_value(benchmark: str, metric: str) -> tuple[str | None, float | None]:
    if benchmark == "mhalubench":
        claim_row = PAPER_TABLES["mhalubench"]["comparison_row"]["claim"]
        segment_row = PAPER_TABLES["mhalubench"]["comparison_row"]["segment"]
        claim_metric = metric.removeprefix("claim_")
        segment_metric = metric.removeprefix("segment_")
        if metric.startswith("claim_"):
            return claim_row, PAPER_TABLES["mhalubench"]["rows"][claim_row].get(claim_metric)
        if metric.startswith("segment_"):
            return segment_row, PAPER_TABLES["mhalubench"]["rows"][segment_row].get(segment_metric)
    if benchmark == "mfhallubench":
        row = PAPER_TABLES["mfhallubench"]["comparison_row"]
        return row, PAPER_TABLES["mfhallubench"]["rows"][row].get(metric)
    if benchmark in {"pope_adv", "llava_bench_wild", "mmhal_bench", "object_halbench", "amber"}:
        row = mitigation_reference_row()
        key = "pope_adv_f1" if benchmark == "pope_adv" and metric == "f1" else metric
        value = PAPER_TABLES["mitigation"]["rows"][row].get(key)
        return row, value
    if benchmark == "hss":
        row = "LLaVA w/ HSA-DPO"
        appendix = {
            "avg_hss": 0.602,
            "sum_hss": 33.50,
            "avg_response_length": 466.8,
        }
        key = "avg_hss" if metric == "hss" else metric
        return row, appendix.get(key)
    return None, None


def paper_base_value(benchmark: str, metric: str) -> tuple[str | None, float | None]:
    if benchmark in {"pope_adv", "llava_bench_wild", "mmhal_bench", "object_halbench", "amber"}:
        row = mitigation_base_row()
        key = "pope_adv_f1" if benchmark == "pope_adv" and metric == "f1" else metric
        value = PAPER_TABLES["mitigation"]["rows"][row].get(key)
        return row, value
    if benchmark == "hss":
        row = "LLaVA-1.5-13b"
        appendix = {
            "avg_hss": 0.796,
            "sum_hss": 42.99,
            "avg_response_length": 579.2,
        }
        key = "avg_hss" if metric == "hss" else metric
        return row, appendix.get(key)
    return None, None
