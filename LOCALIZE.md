# FK-001: FOREIGN KEY constraint failed in code_graph edges

- **File:** `src/superlocalmemory/code_graph/database.py:205`
- **Anomaly:** `sqlite3.IntegrityError: FOREIGN KEY constraint failed` при `INSERT INTO graph_edges` с `__unresolved__` / `__call__*` source/target node_id
- **Condition:** `build_code_graph` для любого репозитория с cross-file вызовами (>1 файла)
- **Root cause:** `schema_code_graph.py:61-62` — `source_node_id` и `target_node_id` имеют `REFERENCES graph_nodes(node_id) ON DELETE CASCADE`. Экстракторы генерируют edges с плейсхолдерами `__unresolved__`/`__call__*`, которые не существуют как `node_id`. FK constraint не даёт их вставить, транзакция откатывается.
- **Fix:** Убрать FK из `graph_edges`, добавить `cleanup_orphaned_edges()`, интегрировать `ImportResolver` для замены `__call__*` на реальные node_id, фильтровать оставшиеся placeholder edges перед записью.
- **Test:** `build_code_graph /tmp/test-ts` — должен вернуть `success`, не FK error
- **Boundaries:** Не затрагивает `graph_nodes`, `code_memory_links` (FK там корректны). Не затрагивает update_code_graph (там фильтр placeholder edges).
- **Blast radius:** `graph_store.py`, `graph_engine.py`, `database.py`, `tools_code_graph.py` (все 11 файлов изменены)
