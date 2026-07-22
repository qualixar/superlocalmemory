# V3 Mathematical Foundations

SuperLocalMemory V3 introduces three mathematical pillars — to our knowledge the first application of each of these techniques to agent memory retrieval, as documented in our public arXiv preprints ([arXiv:2603.14588](https://arxiv.org/abs/2603.14588) | [Zenodo](https://zenodo.org/records/19038659)).

## Published LoCoMo Evidence Carried into V3.7

The published V3 architecture results remain the evidence base for V3.7. They
retain their original protocols and are not a newly rerun V3.7 package result:

| Configuration | Result | Published scope |
|---|---:|---|
| **Mode A Raw** | **60.4%** | 10 conversations / 1,276 questions; local embeddings, local retrieval, zero-LLM answer construction |
| **Mode A Retrieval** | **74.8%** | 10 conversations / 1,276 questions; local retrieval with GPT-4.1-mini answer synthesis |
| **Mode C** | **87.7%** | Conv-30 / 81 questions; cloud embeddings with GPT-4.1-mini answer generation and judge |

The six-conversation mathematics ablation below is a separate protocol. It
reports **71.7%** with information-geometric layers versus **58.9%** without
them (**+12.7pp**) and must not be substituted for the aggregate rows above.

---

## 1. Fisher-Rao Information Geometry

**The problem:** Cosine similarity treats embeddings as direction vectors. Two memories with the same meaning but different confidence look identical.

**Our solution:** We use the Fisher-Rao geodesic distance — the natural metric on statistical manifolds. Each memory embedding is modeled as a diagonal Gaussian distribution with learned mean and variance. Distance is measured along the geodesic (shortest path on the manifold), not through Euclidean space.

> **Shipping default:** SLM ships `fisher_mode="simplified"` — a variance-weighted (Mahalanobis-style) approximation that is fast and stable for the common case. Set `fisher_mode="full"` to activate the full Atkinson-Mitchell geodesic (`arccosh` form) described here.

**What this means in practice:**
- A high-confidence memory and a low-confidence memory about the same topic are distinguished
- Retrieval improves as the system learns — variance shrinks with repeated access (Bayesian conjugate update after 3+ accesses)
- Graduated ramp from cosine to full Fisher-Rao distance over the first 10 accesses per memory
- Computation complexity: Theta(d) time — same order as cosine similarity

**Benchmark impact:** Removing the Fisher metric causes **-10.8pp** on conv-30 ablation. Across 6 conversations (n=832 questions), the three mathematical layers collectively contribute **+12.7pp average improvement**, reaching **+19.9pp on the most challenging dialogues** (conv-44).

**Code:** `src/superlocalmemory/math/fisher.py`

---

## 2. Sheaf Cohomology for Memory Consistency

**The problem:** As memories accumulate, contradictions emerge. "Alice moved to London in March" vs "Alice lives in Paris as of April." Pairwise checking is O(n²) and misses transitive contradictions.

**Our solution:** We model the knowledge graph as a cellular sheaf — an algebraic structure from topology. Each edge carries a restriction map that relates adjacent memories. Computing the first cohomology group H¹(G,F) reveals global inconsistencies:

- **H¹ = 0** — All memories are globally consistent
- **H¹ ≠ 0** — Contradictions exist, even if every local pair looks fine

This catches contradictions that no pairwise method can detect. Runs in **O(|E| * d)** time — subquadratic in N when the context graph is sparse.

**What this means in practice:**
- The system detects when new information contradicts existing knowledge
- Contradictions are flagged with severity scores (>0.45 threshold triggers a SUPERSEDES edge)
- Knowledge graph maintains algebraic consistency as memories accumulate

**Benchmark impact:** Removing sheaf consistency causes **-1.7pp** on single-conversation ablation. The effect is subtle on individual conversations but becomes critical at scale (at N=100,000 memories, expected contradiction count exceeds ~5,000).

**Code:** `src/superlocalmemory/math/sheaf.py`

---

## 3. Riemannian Langevin Dynamics for Memory Lifecycle

**The problem:** Memory systems need lifecycle management — old, unused memories should be archived. Current systems use hardcoded thresholds (e.g., "archive after 30 days"). This doesn't adapt to usage patterns.

**Our solution:** Memory lifecycle evolves via stochastic gradient flow on a Riemannian manifold. The potential function encodes access frequency, trust score, and recency. The dynamics provably converge to a stationary distribution — the mathematically optimal allocation of memories across lifecycle states.

**Four lifecycle states:**
- **Active** — Frequently used, instantly available
- **Warm** — Recently used, included in searches
- **Cold** — Older, retrievable on demand
- **Archived** — Compressed, restorable when needed

**What this means in practice:**
- No manual thresholds — the system self-organizes
- Frequently accessed memories stay active longer
- Low-trust memories decay faster (coupled with Fisher-Rao via information geometry)
- Mathematically guaranteed convergence — not heuristic

**Code:** `src/superlocalmemory/dynamics/fisher_langevin_coupling.py`

---

## Ablation Results

Evaluated on LoCoMo conv-30 (81 scored questions). Each row removes one component.

> **Protocol:** zero-LLM answer construction (same as the Mode A Raw benchmark). This is why the full-system score here (60.4%) differs from the Fisher-Rao table below, which uses local retrieval + GPT-4.1-mini answer synthesis (Mode A Retrieval). The two tables are internally consistent (identical Fisher deltas) but use different answer-construction protocols.

| Configuration | Aggregate (%) | Delta (pp) |
|:-------------|:-----:|:------:|
| **Full system** | **60.4** | — |
| − Fisher metric | 49.6 | **−10.8** |
| − Sheaf consistency | 58.7 | −1.7 |
| − All math layers | 52.8 | **−7.6** |
| − BM25 channel | 53.9 | −6.5 |
| − Entity graph | 59.4 | −1.0 |
| − Temporal channel | 60.2 | −0.2 |
| − Cross-encoder | 29.7 | **−30.7** |

**Key findings:**
- Cross-encoder reranking is the single largest contributor (**−30.7pp** when removed)
- Fisher-Rao metric alone: **−10.8pp** — the largest single mathematical layer effect
- All three math layers collectively: **−7.6pp**
- Bootstrap 95% CI for full system: [53.4, 74.0]; for cross-encoder removed: [17.1, 45.7] — non-overlapping, confirming statistical significance

### Fisher-Rao vs Cosine (6 Conversations, n=832)

> **Protocol:** local retrieval + GPT-4.1-mini answer synthesis (Mode A Retrieval) — not directly comparable to the zero-LLM ablation table above.

| Conversation | With Math (%) | Without Math (%) | Delta (pp) |
|:-------------|:-----:|:-------:|:-----:|
| Easiest (conv-26) | 78.5 | 71.2 | +7.3 |
| conv-30 | 77.5 | 66.7 | +10.8 |
| conv-42 | 60.8 | 47.3 | +13.5 |
| conv-43 | 64.3 | 58.3 | +6.0 |
| Hardest (conv-44) | 64.2 | 44.3 | **+19.9** |
| conv-49 | 84.7 | 65.9 | +18.8 |
| **Average** | **71.7** | **58.9** | **+12.7** |

Mathematical layers provide the greatest benefit precisely where heuristic methods struggle — the harder the conversation, the bigger the improvement.

---

## Why These Specific Methods?

| Method | Why We Chose It | Alternative We Rejected |
|--------|----------------|------------------------|
| Fisher-Rao | Natural metric for probability distributions; captures uncertainty | Cosine similarity (ignores confidence) |
| Sheaf Cohomology | Detects global inconsistencies from local data; scales algebraically | Pairwise contradiction checking (O(n²), misses transitive) |
| Riemannian Langevin | Provable convergence; couples naturally with Fisher metric | Hardcoded thresholds (doesn't adapt) |

---

## Research Paper

For the full mathematical treatment including proofs, theorems, and detailed experimental methodology:

**SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory**

*Varun Pratap Bhardwaj, Independent Researcher, 2026*

[arXiv:2603.14588](https://arxiv.org/abs/2603.14588) | [Zenodo DOI: 10.5281/zenodo.19038659](https://zenodo.org/records/19038659)

```bibtex
@article{bhardwaj2026slmv3,
  title={Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.14588},
  year={2026},
  url={https://arxiv.org/abs/2603.14588}
}
```

---

*Part of [Qualixar](https://qualixar.com) · Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
