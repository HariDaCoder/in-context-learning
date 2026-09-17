"""Small, conservative audit gate for expensive BO experiment groups."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("results/bo_matrix/scientific_audit")
MAIN_GROUPS = frozenset(("matched_rho", "dimension", "architecture"))
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _metric(record, name):
    value = record.get("metrics", {}).get(name, {})
    return value.get("mean") if isinstance(value, dict) else None


def _finite_positive(value):
    return isinstance(value, (int, float)) and value > 0


def audit(inputs, output_dir=DEFAULT_OUTPUT_DIR):
    """Audit existing JSON result bundles and write JSON, CSV, and Markdown.

    A missing nonempty, fully-matched Transformer BO region deliberately
    produces STOP_AND_REVISE_BO_DEFINITION.  This is a gate, not a claim that
    benign overfitting is absent in general.
    """

    rows = []
    for input_path in inputs:
        path = Path(input_path)
        document = json.loads(path.read_text(encoding="utf-8"))
        documents = [(path, document)]
        for job in document.get("jobs", []):
            result_path = job.get("result_path") if isinstance(job, dict) else None
            if result_path and job.get("kind") == "checkpoint":
                candidate = REPOSITORY_ROOT / result_path
                if candidate.is_file():
                    documents.append((candidate, json.loads(candidate.read_text(encoding="utf-8"))))
        for source, result in documents:
            for record in result.get("records", []):
                if not isinstance(record, dict) or record.get("checkpoint_id") is None:
                    continue
                rows.append({
                "source": str(source),
                "checkpoint_id": record.get("checkpoint_id"),
                "train_regime": record.get("train_distribution", {}).get("source", "unknown"),
                "dependence_protocol": record.get("dependence_protocol", record.get("protocol")),
                "snr_protocol": record.get("snr_protocol", "unknown"),
                "context_protocol": record.get("context_protocol", "unknown"),
                "fully_matched": record.get("fully_matched", False),
                "direct_bo_frequency": _metric(record, "direct_bo_candidate"),
                "linear_bo_frequency": _metric(record, "linear_bo_candidate"),
                "clean_gen_ratio": _metric(record, "clean_gen_ratio"),
                "linear_clean_gen_ratio": _metric(record, "linear_clean_gen_ratio"),
                "heldout_probe_r2": _metric(record, "heldout_probe_r2"),
                "duplicate_fit_ratio": _metric(record, "duplicate_fit_ratio"),
                    "linear_fit_ratio": _metric(record, "linear_fit_ratio"),
                })
    fully_matched = [row for row in rows if row["fully_matched"]]
    direct_region = [row for row in fully_matched if _finite_positive(row["direct_bo_frequency"])]
    linear_region = [row for row in fully_matched if _finite_positive(row["linear_bo_frequency"])]
    recommendation = "PROCEED_MAIN_BO" if direct_region or linear_region else "STOP_AND_REVISE_BO_DEFINITION"
    reasons = []
    if not rows:
        reasons.append("No Transformer evaluation rows were supplied.")
    if rows and not fully_matched:
        reasons.append("No rows are fully matched across dependence, SNR, and context support.")
    if fully_matched and not (direct_region or linear_region):
        reasons.append("No nonempty normalized direct or linear BO region was observed.")
    if not reasons:
        reasons.append("At least one fully matched normalized BO candidate region was observed.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "created_at": created,
        "recommendation": recommendation,
        "inputs": [str(Path(item)) for item in inputs],
        "transformer_rows": len(rows),
        "fully_matched_rows": len(fully_matched),
        "direct_bo_rows": len(direct_region),
        "linear_bo_rows": len(linear_region),
        "reasons": reasons,
        "rows": rows,
    }
    json_path = output / "scientific_audit.json"
    csv_path = output / "scientific_audit.csv"
    md_path = output / "scientific_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = tuple(rows[0]) if rows else ("source", "checkpoint_id", "fully_matched")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(
        "# Scientific audit\n\n"
        "Recommendation: **{}**\n\n"
        "Transformer rows: {}; fully matched rows: {}; direct BO rows: {}; linear BO rows: {}.\n\n"
        "{}\n".format(recommendation, len(rows), len(fully_matched), len(direct_region), len(linear_region), " ".join(reasons)),
        encoding="utf-8",
    )
    return report, json_path, csv_path, md_path


def require_main_gate(repository_root):
    path = Path(repository_root) / DEFAULT_OUTPUT_DIR / "scientific_audit.json"
    if not path.is_file():
        raise RuntimeError("main BO matrix is gated: run scientific-audit after the matched-SNR pilot")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("recommendation") != "PROCEED_MAIN_BO":
        raise RuntimeError("main BO matrix is gated by scientific audit: {}".format(document.get("recommendation")))
    return path
