from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    return REPO_ROOT


def dump_json(path: str | Path, payload: Any) -> Path:
    out = ensure_parent_dir(path)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
    return out


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_specs(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("models manifest must be a JSON list")
    return [dict(item) for item in payload]


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def quantize_float(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def ece_score(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    bucket_totals = [0] * bins
    bucket_conf = [0.0] * bins
    bucket_acc = [0.0] * bins
    for label, prob in zip(labels, probabilities):
        idx = min(bins - 1, max(0, int(prob * bins)))
        bucket_totals[idx] += 1
        bucket_conf[idx] += prob
        bucket_acc[idx] += float(label)
    total = len(labels)
    if total == 0:
        return None
    error = 0.0
    for count, conf_sum, acc_sum in zip(bucket_totals, bucket_conf, bucket_acc):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = acc_sum / count
        error += (count / total) * abs(avg_conf - avg_acc)
    return float(error)


def binary_classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[float] | None = None,
) -> dict[str, float | None]:
    total = len(labels)
    if total == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "nll": None,
            "ece": None,
        }

    tp = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 1)
    fp = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 1)
    fn = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 0)
    tn = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = safe_div(tp + tn, total)

    auroc = None
    auprc = None
    brier = None
    nll = None
    ece = None
    if scores is not None and len(set(labels)) > 1:
        clipped = [min(1.0 - 1e-6, max(1e-6, float(score))) for score in scores]
        auroc = float(roc_auc_score(labels, clipped))
        auprc = float(average_precision_score(labels, clipped))
        brier = float(brier_score_loss(labels, clipped))
        nll = float(log_loss(labels, clipped))
        ece = ece_score(labels, clipped)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "nll": nll,
        "ece": ece,
    }


def macro_f1_from_confusion(confusion: Mapping[str, Mapping[str, int]]) -> float | None:
    labels = sorted({*confusion.keys(), *[k for row in confusion.values() for k in row]})
    if not labels:
        return None
    f1s: list[float] = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in labels if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision is None or recall is None or (precision + recall) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len((text or "").split())


def default_dataset_root(dataset_root_override: str | None, name: str) -> str:
    if dataset_root_override:
        return str(Path(dataset_root_override) / name)
    return str(repo_root() / "playground" / "data" / "eval" / name)


def resolve_existing(*candidates: str | Path | None) -> str | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def discover_stage_paths() -> dict[str, str]:
    output_root = repo_root() / "output" / "fghd"
    candidates = {
        "stage3_det": [
            output_root / "D_det_calibrated.jsonl",
            output_root / "D_det.jsonl",
        ],
        "stage3_report": [
            output_root / "D_det_calibration.json",
        ],
        "stage4_rewrite": [
            output_root / "D_rewrite_grouped.jsonl",
            output_root / "D_rewrite.jsonl",
        ],
        "stage5_pref": [
            output_root / "D_pref_clean_grouped.jsonl",
            output_root / "D_pref_clean.jsonl",
        ],
        "stage5_tau": [
            output_root / "D_tau_c_report_grouped.json",
            output_root / "D_tau_c_report.json",
        ],
        "stage6_dir": [
            output_root / "adaptive_dpo",
        ],
    }
    discovered: dict[str, str] = {}
    for key, values in candidates.items():
        path = resolve_existing(*values)
        if path:
            discovered[key] = path
    return discovered


def summarize_stage3(path: str | Path, calibration_report: str | Path | None = None) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    total_rows = len(rows)
    signals = [signal for row in rows for signal in row.get("signals", [])]
    real_signals = [
        signal
        for signal in signals
        if not (signal.get("metadata") or {}).get("is_placeholder")
        and not (signal.get("metadata") or {}).get("error")
    ]
    labels = [0 if (row.get("raw_detection") or "").strip().upper() == "NO HALLUCINATION" else 1 for row in rows]
    scores = [
        max([float(signal.get("confidence", 0.0)) for signal in row.get("signals", [])], default=0.0)
        for row in rows
    ]
    predictions = [1 if score > 0.5 else 0 for score in scores]
    summary = binary_classification_metrics(labels, predictions, scores=scores)
    per_type: dict[str, int] = {}
    per_severity: dict[str, int] = {}
    for signal in real_signals:
        per_type[signal.get("hallucination_type") or "unknown"] = (
            per_type.get(signal.get("hallucination_type") or "unknown", 0) + 1
        )
        per_severity[str(signal.get("severity"))] = per_severity.get(str(signal.get("severity")), 0) + 1
    result: dict[str, Any] = {
        "rows": total_rows,
        "signals": len(signals),
        "real_signals": len(real_signals),
        "no_hallucination_rows": sum(1 for label in labels if label == 0),
        "per_type": per_type,
        "per_severity": per_severity,
        "row_accuracy": summary["accuracy"],
        "row_precision": summary["precision"],
        "row_recall": summary["recall"],
        "row_f1": summary["f1"],
        "auroc": summary["auroc"],
        "auprc": summary["auprc"],
        "brier": summary["brier"],
        "nll": summary["nll"],
        "ece": summary["ece"],
    }
    if calibration_report and Path(calibration_report).exists():
        report = load_json(calibration_report)
        policy = report.get("group_threshold_policy", {})
        result["threshold_policy"] = {
            "global_threshold": policy.get("global_threshold"),
            "group_count": len(policy.get("by_group", {})),
            "low_support_groups": sum(
                1
                for value in (policy.get("by_group", {}) or {}).values()
                if isinstance(value, Mapping) and value.get("insufficient_support")
            ),
        }
    return result


def summarize_stage4(path: str | Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    backend_counts: dict[str, int] = {}
    filtered_counts: list[int] = []
    type_counts: dict[str, int] = {}
    generated = 0
    skipped = 0
    for row in rows:
        metadata = row.get("metadata") or {}
        backend = metadata.get("rewrite_backend", "unknown")
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        status = metadata.get("rewrite_status")
        if status == "skipped_no_reliable_signals":
            skipped += 1
        else:
            generated += 1
        filtered = row.get("filtered_signals", []) or []
        filtered_counts.append(len(filtered))
        for signal in filtered:
            key = signal.get("hallucination_type") or "unknown"
            type_counts[key] = type_counts.get(key, 0) + 1
    return {
        "rows": len(rows),
        "generated": generated,
        "skipped": skipped,
        "rewrite_backend_distribution": backend_counts,
        "avg_filtered_signals": mean_or_none(filtered_counts),
        "per_type_filtered_signals": type_counts,
        "uses_group_threshold_policy": any(
            (row.get("metadata") or {}).get("threshold_policy") == "group_conditional"
            for row in rows
        ),
    }


def summarize_stage5(pref_path: str | Path, tau_report: str | Path | None = None) -> dict[str, Any]:
    rows = list(read_jsonl(pref_path))
    pair_confidences = [float(row.get("pair_confidence", 0.0)) for row in rows]
    severity_weights = [float(row.get("severity_weight", 0.0)) for row in rows]
    adaptive_weights = [float(row.get("adaptive_weight", 0.0)) for row in rows]
    summary = {
        "kept_rows": len(rows),
        "pair_confidence_mean": mean_or_none(pair_confidences),
        "pair_confidence_min": min(pair_confidences) if pair_confidences else None,
        "pair_confidence_max": max(pair_confidences) if pair_confidences else None,
        "severity_weight_mean": mean_or_none(severity_weights),
        "adaptive_weight_mean": mean_or_none(adaptive_weights),
    }
    if tau_report and Path(tau_report).exists():
        report = load_json(tau_report)
        summary["selected_tau_c"] = report.get("selected_tau_c")
        summary["tau_c_method"] = report.get("method")
    return summary


def summarize_stage6(stage6_dir: str | Path) -> dict[str, Any]:
    stage6_path = Path(stage6_dir)
    trainer_state = resolve_existing(
        stage6_path / "trainer_state.json",
        *stage6_path.rglob("trainer_state.json"),
    )
    result: dict[str, Any] = {"stage6_dir": str(stage6_path)}
    if not trainer_state:
        return result
    payload = load_json(trainer_state)
    log_history = payload.get("log_history", []) or []
    train_logs = [entry for entry in log_history if "loss" in entry]
    eval_logs = [entry for entry in log_history if any(key.startswith("eval_") for key in entry)]
    if train_logs:
        result["final_train_loss"] = train_logs[-1].get("loss")
    if eval_logs:
        latest = eval_logs[-1]
        for key in (
            "eval_rewards/accuracies",
            "eval_rewards/margins",
            "eval_adaptive/weight_mean",
            "eval_adaptive/pair_confidence_mean",
            "eval_adaptive/severity_weight_mean",
        ):
            if key in latest:
                result[key] = latest[key]
    return result


def getenv_openai_judge_model(default: str | None = None) -> str | None:
    return os.environ.get("OPENAI_JUDGE_MODEL", default)
