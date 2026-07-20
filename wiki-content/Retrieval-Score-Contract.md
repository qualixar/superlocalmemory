# Retrieval Score Contract v2

Retrieval ordering is not answer confidence. V3.7 exposes separate fields so a
client does not mistake a ranking value for a probability.

| Field | Meaning |
|---|---|
| `relevance_score` | Query-relative relevance, bounded to `0.0..1.0` |
| `ranking_score` | Internal ranking utility; diagnostic, not a probability |
| `memory_confidence` | Confidence stored with the underlying assertion |
| `trust_score` | Evidence-policy trust signal |
| `rank_position` | One-based returned position |

For one compatibility release, `score` aliases `relevance_score` and
`confidence` aliases `memory_confidence`.

Canonical recall responses declare:

```json
{
  "score_contract_version": "2",
  "calibration_status": "uncalibrated",
  "calibration_id": null,
  "answer_confidence": null
}
```

The current release does not claim calibrated answer confidence. Do not derive
one by transforming or averaging retrieval scores. Empty result sets declare
an abstention reason (`evidence_floor` or `no_candidates`).

Current candidate producers are dense semantic, BM25 lexical, temporal,
Hopfield associative, and spreading activation. Entity-graph information is a
post-fusion score enhancement in the current implementation, not a sixth
candidate producer. Optional reranking and adaptive learning may affect
ordering without changing the meaning of the public score fields.

The complete developer contract and calibration gate are in
[`docs/retrieval-score-contract.md`](https://github.com/qualixar/superlocalmemory/blob/main/docs/retrieval-score-contract.md).
