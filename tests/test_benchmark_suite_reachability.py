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

    exp_files = sorted(p.name for p in exp_dir.glob("exp*.py"))
    assert exp_files, "no exp*.py files found"

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
