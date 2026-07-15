# Modes Explained

SuperLocalMemory V3 offers three operating modes. Choose based on your privacy requirements and accuracy needs.

## Mode A: Local Guardian

**Local core processing with explicit optional integrations.**

- Core memory state uses the configured local data root
- No model-provider call is required for core memory operations; connectors, backup, downloads, and other enabled integrations have separate network behavior
- Retrieval can use dense semantic, BM25 lexical, temporal, Hopfield
  associative, and spreading-activation candidate producers
- Optional reranking and entity-graph score enhancement depend on runtime health
- Fisher, sheaf, and Langevin components can operate without a cloud LLM; this
  page makes no universal accuracy claim
- Local storage, erasure, provenance, policy, and audit controls are available for deployment-specific assessment

**Who it's for:** Privacy-conscious developers, enterprise environments with strict data policies, EU-regulated industries, air-gapped systems.

**Limitations:** No LLM-powered answer synthesis. Returns ranked memory excerpts rather than composed answers. Best accuracy on factual and entity-based queries.

**Historical V3 research result:** 74.8% used local retrieval with GPT-4.1-mini answer construction. It is not an end-to-end zero-LLM or current V3.7 result.

## Mode B: Smart Local

**Local LLM through an operator-managed endpoint.**

- Everything in Mode A, plus a local LLM via Ollama
- The LLM synthesizes retrieved memories into coherent answers
- Uses an operator-configured Ollama endpoint; confirm where it runs and which optional integrations are enabled
- Requires Ollama installed with a model (e.g., `llama3.2`, `mistral`, `phi3`)

**Who it's for:** Developers who want composed answers but need data to stay local. Teams that can run Ollama on their machines.

**Requirements:**
- [Ollama](https://ollama.com/) installed
- At least one model pulled: `ollama pull llama3.2`
- 8GB+ RAM recommended for good model performance

**Limitations:** Answer quality depends on the local model's capabilities. Smaller models may produce less accurate synthesis.

## Mode C: Full Power

**Maximum accuracy. Cloud LLM for fact extraction and answer synthesis.**

- Everything in Mode B, plus cloud LLM support
- LLM-powered fact extraction for richer ingestion
- Agentic retrieval with multi-round refinement
- Supports Azure OpenAI, OpenAI, Anthropic, and other providers

**Who it's for:** Developers who prioritize accuracy over privacy. Teams with approved cloud AI policies. Research and benchmarking.

**Historical Mode C result:** 87.7% on 81 questions from one conversation with cloud-assisted components. It is not a full-dataset or current V3.7 score.

**Note:** Data is sent to the cloud provider you configure. Ensure your organization's policies allow this.

## Switching Modes

Check your current mode:

```bash
slm mode
```

Switch modes:

```bash
slm mode a    # Switch to Local Guardian
slm mode b    # Switch to Smart Local
slm mode c    # Switch to Full Power
```

Mode changes take effect immediately. Your stored memories are not affected — all modes use the same database.

## Comparison Table

| Feature | Mode A | Mode B | Mode C |
|---------|:------:|:------:|:------:|
| Semantic search | Yes | Yes | Yes |
| Keyword search (BM25) | Yes | Yes | Yes |
| Entity graph | Yes | Yes | Yes |
| Temporal retrieval | Yes | Yes | Yes |
| Mathematical scoring | Yes | Yes | Yes |
| Cross-encoder reranking | Yes | Yes | Yes |
| LLM fact extraction | No | Local | Cloud |
| LLM answer synthesis | No | Local | Cloud |
| Agentic retrieval | No | No | Yes |
| Provider call in core memory path | No | Local Ollama endpoint | Cloud provider |
| Compliance status | Deployment assessment required | Deployment assessment required | Deployment and provider assessment required |
| Internet required | No | No | Yes |

## Recommendations

- **Start with Mode A** if you are unsure. You can always upgrade later.
- **Use Mode B** if you have a capable machine (16GB+ RAM) and want composed answers locally.
- **Use Mode C** for maximum accuracy when cloud access is acceptable.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
