"""Invariant F-06: every exp*.py on disk is reachable from run_all.py.

DISCOVERED FROM DISK — do not hardcode a list. This makes "paper cites an
experiment that never ran" impossible to repeat. MUST fail against the
pre-F-06 run_all.py (which lacked exp2b + governed latency wiring).
"""
import ast
import pathlib


def test_every_exp_reachable_from_run_all():
    """Every exp*.py must be BOTH imported by AND invoked from run_all.py.

    Reachability is decided on the AST only. An earlier version of this test
    fell back to ``mod not in run_all_source`` — a raw substring check against
    the file text — which a *comment* mentioning the module name was enough to
    satisfy. Verified 2026-08-08: adding an unwired ``expZZZ_dummy.py`` plus the
    line ``# NOTE: expZZZ_dummy is intentionally not wired up yet.`` made this
    test PASS while the F-06 defect was fully present. A gate that a comment can
    turn green is not a gate. Do not reintroduce any text-level matching here.

    Import alone is also insufficient: a module can be imported and never
    called, which reproduces F-06 (cited by the paper, absent from the bundle).
    """
    # Gates live in tests/ so CI collects them (pyproject testpaths = ["tests"]);
    # the harness they guard lives in benchmark/. Resolve relative to the repo
    # root — never hardcode an absolute path in a published repository.
    exp_dir = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
    tree = ast.parse((exp_dir / "run_all.py").read_text(encoding="utf-8"))

    # module name -> the local name it is bound to (alias, or the module itself)
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.name] = alias.asname or alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            bound.setdefault(node.module, node.module)

    # local names used as the receiver of a call, e.g. `exp2b.run(...)`
    invoked: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            if isinstance(recv, ast.Name):
                invoked.add(recv.id)

    # Glob every benchmark module, not just `exp*.py`. The earlier `exp*.py`
    # pattern missed `bench_perf.py` — which is orphaned from run_all.py and yet
    # produces bench_perf.json, the sole evidence for the paper's entire
    # Real-Scale Performance section. A gate written around a naming convention
    # only guards files that happen to follow it (F-22).
    _INFRASTRUCTURE = {"run_all.py", "_harness.py", "__init__.py", "conftest.py"}

    # Modules that legitimately cannot run in the default bundle. Each MUST carry
    # a reason. Being listed here is a declaration, not an exemption from
    # scrutiny: the paper may only cite one of these alongside the stated
    # precondition under which it was produced.
    _MANUAL_RUN = {
        "bench_perf.py": (
            "requires a retained ~1 GB real memory-store database (--db); "
            "cannot run in a fresh-temp-store bundle"
        ),
    }

    exp_files = sorted(
        p.name
        for p in exp_dir.glob("*.py")
        if p.name not in _INFRASTRUCTURE and not p.name.startswith("test_")
    )
    assert exp_files, "no benchmark modules found"

    undeclared_manual = set(_MANUAL_RUN) - set(exp_files)
    assert not undeclared_manual, (
        f"_MANUAL_RUN names modules that no longer exist: {sorted(undeclared_manual)}"
    )
    exp_files = [f for f in exp_files if f not in _MANUAL_RUN]

    unreachable = []
    for fname in exp_files:
        mod = fname[:-3]
        local = bound.get(mod)
        if local is None:
            unreachable.append(f"{fname} (never imported)")
        elif local not in invoked:
            unreachable.append(f"{fname} (imported as '{local}' but never called)")

    assert not unreachable, (
        "Experiments unreachable from run_all.py — the paper must never cite an "
        f"experiment the runner does not run: {unreachable}\n"
        f"imported={sorted(bound)}\ninvoked={sorted(invoked)}\nfiles={exp_files}"
    )


if __name__ == "__main__":
    test_every_exp_reachable_from_run_all()
    print("PASS F-06 reachability")
