# FK-001: Acceptance Criteria

## Bug: FOREIGN KEY constraint failed in code_graph edges

### Acceptance Criteria

- [x] Bug воспроизводится до патча (reproducer)
  - `build_code_graph /tmp/test-ts` до патча → `FOREIGN KEY constraint failed`
- [x] Bug не воспроизводится после патча
  - `build_code_graph /tmp/test-ts` после патча → `success: true`, 12124 nodes, 11236 edges
- [x] Все существующие тесты проходят
  - pytest phase1: 56/56 passed
  - syntax check: 11 files OK
  - ruff: 0 new errors (731 pre-existing, all formatting)
- [x] parallel remember не блокируется (SLM-001)
  - `remember` за 0.1s во время `build_code_graph`
- [x] import_maps pipeline работает
  - Экстрактор → parser → resolver — данные проходят через все слои
- [x] cleanup orphaned edges работает
  - После 3+ `build_code_graph` вызовов — orphaned edges: 0
- [x] OPE: свежие данные корректны
  - orphans: 0, ping: 0.001s, embedding_cpu: 0%, parallel_ok: 1, tasks: 0

### Решение

PASS — баг устранён, регрессий нет, OPE чисто. PR готов к ревью.
