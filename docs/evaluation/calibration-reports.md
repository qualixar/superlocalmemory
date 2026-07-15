# Held-out calibration reports

SLM's retrieval score is a ranking utility, not a probability that an answer is
correct. The calibration report tool therefore requires two separate fields:

- `raw_retrieval_score` is retained only for distribution diagnostics.
- `predicted_probability` must come from an explicitly named, separately fitted
  calibrator. It is the only value used for Brier score, ECE, reliability bins,
  and selective-risk calculations.

The tool does not fit a calibrator and does not activate one in production. A
successful report remains `runtime_calibration_status: "uncalibrated"` and
`runtime_activation_allowed: false`. This prevents a local report, synthetic
fixture, or raw retrieval score from silently becoming public answer
confidence.

## Input contract

The input uses schema `superlocalmemory.calibration-input/v1` and must contain:

```json
{
  "schema": "superlocalmemory.calibration-input/v1",
  "calibrator": {
    "calibration_id": "frozen-calibrator-id",
    "method": "isotonic_regression",
    "fit_dataset_id": "calibration-train-id",
    "fit_split": "calibration_train",
    "fit_artifact_sha256": "64 lowercase hex characters"
  },
  "evaluation": {
    "dataset_id": "different-held-out-id",
    "split": "held_out",
    "examples_sha256": "SHA-256 of the canonical examples array"
  },
  "examples": [
    {
      "example_id": "unique-query-id",
      "raw_retrieval_score": 0.73,
      "predicted_probability": 0.61,
      "label": 1
    }
  ]
}
```

The displayed array is structural only. A report requires at least 100 unique
held-out examples. The fitting and evaluation dataset IDs must differ, the
evaluation split must be `held_out`, both hashes must be present, labels must be
binary, and predicted probabilities must be finite values in `[0, 1]`.
Identity mappings such as `method: "raw_score"` are rejected.

`examples_sha256` is calculated from compact, key-sorted UTF-8 JSON:

```python
json.dumps(examples, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

## Generate a report

```bash
python3 scripts/calibration_report.py \
  --input /path/to/held-out-input.json \
  --output /path/to/calibration-report.json
```

The output uses schema `superlocalmemory.calibration-report/v1` and includes:

- Brier score, equal-width ECE, and maximum calibration error.
- Reliability bins with count, mean predicted probability, observed accuracy,
  and absolute gap.
- Selective risk, coverage, and abstention rate at declared thresholds.
- Separate raw-score and probability distributions.
- Calibrator and held-out dataset provenance hashes.
- A non-activating runtime status.

This local tool does not produce LoCoMo evidence and no calibration result is
checked into the repository. Until a real held-out input is supplied, reviewed,
and frozen through a separate runtime activation contract, SLM must continue to
return `answer_confidence: null` and `calibration_status: "uncalibrated"`.
