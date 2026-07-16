# Retrieval Score Contract v2

This contract describes runtime result fields, not benchmark accuracy. For the
published V3 LoCoMo figures carried into V3.7 and their protocol disclosures,
see [Benchmark Evidence](benchmarks.md).

SuperLocalMemory V3.7 keeps retrieval ordering separate from confidence. A
retrieval score answers “how relevant is this stored fact to this query?” It
does not answer “how likely is a generated answer to be correct?”

## Result fields

| Field | Meaning | Public interpretation |
|---|---|---|
| `relevance_score` | Query-relative relevance after the configured retrieval and scoring pipeline | Bounded to `0.0..1.0`; compare only within a compatible SLM release and retrieval configuration |
| `ranking_score` | Internal ranking utility after optional adaptive or reranking adjustments | Diagnostic value; it is not a probability and is not guaranteed to be bounded or comparable across configurations |
| `memory_confidence` | Confidence stored with the underlying assertion | Memory metadata; it is not query relevance or answer correctness |
| `trust_score` | Trust signal attached to the stored evidence | An evidence-policy signal, not answer confidence |
| `rank_position` | One-based position in the returned result list | The observable order after ranking |

For one compatibility release, legacy `score` aliases `relevance_score` and
legacy `confidence` aliases `memory_confidence`. New integrations should use
the explicit field names.

## Response fields

Every canonical recall response declares:

```json
{
  "score_contract_version": "2",
  "calibration_status": "uncalibrated",
  "calibration_id": null,
  "answer_confidence": null,
  "abstained": false,
  "abstention_reason": null
}
```

V3.7 does not publish calibrated answer confidence. Therefore
`calibration_status` is `uncalibrated`, `calibration_id` is `null`, and
`answer_confidence` is `null`. Consumers must not derive answer probability by
doubling, thresholding, averaging, or otherwise transforming retrieval scores.

When no result survives the evidence floor, `abstained` is `true` and
`abstention_reason` is `evidence_floor`. When candidate generation returns
nothing, the reason is `no_candidates`.

## Retrieval composition

The current engine can run five candidate producers when their dependencies
are healthy: dense semantic, BM25 lexical, temporal, Hopfield associative, and
spreading activation. Weighted reciprocal-rank fusion combines their output.
Entity-graph information can enhance a post-fusion score but does not create an
independent candidate in the current implementation. Optional reranking and
adaptive learning can alter `ranking_score`; they do not turn it into a
probability.

The exact channels that contributed to a result are available through trace
output. A missing optional dependency can change the active channel set, so
applications that need a locked retrieval topology should verify health and
trace metadata at startup.

## Calibration release gate

Calibrated confidence requires a frozen release candidate, held-out relevance
labels, declared corpus and query distribution, calibration identity, and
reported calibration and selective-risk metrics. Until that evidence exists,
the truthful contract remains uncalibrated with `answer_confidence: null`.
