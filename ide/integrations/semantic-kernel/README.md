# semantic-kernel-superlocalmemory

[Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/) vector
store (`VectorStoreCollection`) backed by the local data root of
[SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give a Semantic Kernel agent a record store for memory in your local SLM data
root, visible through every other SLM surface (CLI, MCP, dashboard). Optional
SLM providers, connectors, backup, and downloads have separate network behavior.

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `semantic-kernel >= 1.34.0` (the post-1.34 vector-store API)

## Installation

```bash
pip install semantic-kernel-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/semantic-kernel
pip install -e .
```

## Quick start

```python
from dataclasses import dataclass
from typing import Annotated

from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from semantic_kernel_superlocalmemory import SuperLocalMemoryVectorStore


@vectorstoremodel
@dataclass
class Doc:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data")]


store = SuperLocalMemoryVectorStore()
collection = store.get_collection(Doc, collection_name="docs")

await collection.ensure_collection_exists()
await collection.upsert(Doc(id="d1", text="hello world"))
doc = await collection.get("d1")
await collection.delete("d1")
```

## Status and scope

Written against **Semantic Kernel 1.44** (`semantic_kernel.data.vector`).
Semantic Kernel replaced the deprecated `MemoryStoreBase` with the
`VectorStoreCollection` abstraction in the 1.34 (June 2025) Python overhaul;
this adapter targets that current API.

- **CRUD + collection lifecycle** (`upsert` / `get` / `delete` /
  `ensure_collection_exists` / `collection_exists` / `ensure_collection_deleted`)
  delegate to a framework-free record store that is independently tested against
  the SLM engine.
- **Search** (`_inner_search`) performs field-filter retrieval over stored
  records and reports `score=None`; dense-vector ANN ranking is delegated to
  SuperLocalMemory's native recall and is a documented follow-up.
- The SK vector-store API is still marked **preview** — this package pins
  `semantic-kernel>=1.34.0` and should be validated in CI against the installed
  framework version.

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
