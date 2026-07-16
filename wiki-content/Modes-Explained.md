# Operating Modes

SuperLocalMemory has three modes. All retain the local canonical memory store;
the mode selects the model/provider behavior used for configured enrichment and
retrieval work. Switch at any time with `slm mode a`, `slm mode b`, or `slm
mode c`; stored memory is not discarded by a mode change.

| Mode | Core behavior | Model boundary | Best starting point |
|---|---|---|---|
| **A — Local Guardian** | Local memory, retrieval and mathematical/lifecycle components | No cloud model provider is required for core memory operations | Default for a local-first deployment |
| **B — Smart Local** | Mode A plus an operator-configured Ollama endpoint | The selected endpoint and enabled integrations determine the data path | Local model synthesis/enrichment |
| **C — Provider-assisted** | Local memory plus configured provider-backed behavior | Query, ingestion or enrichment content may be sent to the configured provider | Approved external-provider deployment |

## Mode A: Local Guardian

Mode A stores and retrieves through the configured local data root. It can use
semantic, BM25, temporal, Hopfield, and spreading-activation candidates when
their dependencies are healthy, followed by fusion and configured scoring.

“Local core” is not an absolute network claim. Model/dependency downloads,
connectors, cloud backup, proxy providers, and other integrations can use the
network when explicitly enabled. Review the complete deployment before making
a privacy or compliance determination.

## Mode B: Smart Local

Mode B adds a local LLM path through Ollama. The operator chooses the endpoint
and model, then verifies both with `slm doctor` and a representative store /
recall witness. Answer and extraction quality depend on that selected model,
the corpus, and the healthy retrieval channels.

```bash
# Install Ollama using reviewed platform instructions, then:
ollama pull llama3.2
slm mode b
slm doctor
```

## Mode C: Provider-assisted

Mode C enables configured provider-backed enrichment and retrieval behavior
while the canonical memory database remains local. Configure the provider with
`slm provider set` and use environment-managed credentials where appropriate.
The provider’s data-processing terms and the deployment’s network controls
apply to content sent through this mode.

## Compare capabilities accurately

| Capability | A | B | C |
|---|:---:|:---:|:---:|
| Local canonical SQLite memory | Yes | Yes | Yes |
| Multi-channel retrieval | Healthy channels participate | Healthy channels participate | Healthy channels participate |
| Local LLM endpoint | Not required | Configured Ollama | Optional/configured |
| External provider path | Not required for core memory | Not required for core memory | Configured provider path |
| Optional adapters, backup, Mesh or proxy | Explicit operator choice | Explicit operator choice | Explicit operator choice |

Use `slm trace "representative query"` to see the evidence and channels used
by an actual recall rather than inferring behavior from the selected mode.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
