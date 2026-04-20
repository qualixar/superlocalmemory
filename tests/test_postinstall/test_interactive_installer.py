"""Tests for scripts/postinstall-interactive.js — Track D.3 interactive installer.

Per IMPLEMENTATION-MANIFEST-v3.4.22-FINAL.md §D.3 and MASTER-PLAN §5.

The installer is a Node script (invoked by `npm postinstall` on 18,000+ live
npm users, so porting to Python or Bash is not allowed). These Python tests
drive the Node script via `subprocess.run([...])` using deterministic CLI
flags (`--dry-run`, `--profile=<name>`, `--reconfigure`, `--home=<path>`,
`--reply-file=<json>`) so we never require a real TTY.

Contract tests (all 6 from the manifest):
    1. test_tty_detection_defaults_to_balanced_for_non_tty
    2. test_benchmark_recommends_minimal_on_low_ram
    3. test_custom_profile_writes_config_toml
    4. test_reconfigure_preserves_config_bak
    5. test_skill_evolution_prompt_defaults_no
    6. test_no_opus_in_model_choices
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "postinstall-interactive.js"


def _run(
    args: list[str],
    *,
    env: dict | None = None,
    stdin: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Invoke the Node installer with deterministic CLI flags."""
    base_env = os.environ.copy()
    # Force non-TTY for subprocess stdin by default; tests that need TTY
    # simulation inject their own env.
    if env:
        base_env.update(env)
    return subprocess.run(
        ["node", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        input=stdin,
        env=base_env,
        timeout=timeout,
    )


def _read_toml_kv(path: Path) -> dict:
    """Tiny TOML reader — enough to assert on top-level key=value pairs and
    [section] headers. Does NOT implement full TOML; the installer writes a
    flat dialect with section headers and scalar values only.
    """
    result: dict = {}
    section = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            result[section] = {}
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if val in ("true", "false"):
            val = val == "true"
        elif re.fullmatch(r"-?\d+", val):
            val = int(val)
        elif re.fullmatch(r"-?\d+\.\d+", val):
            val = float(val)
        if section is None:
            result[key] = val
        else:
            result[section][key] = val
    return result


# --------------------------------------------------------------------------
# D.3 manifest tests (exact names required — do not rename)
# --------------------------------------------------------------------------


def test_tty_detection_defaults_to_balanced_for_non_tty(tmp_path: Path) -> None:
    """Non-TTY invocations (CI, `CI=true`, piped stdin) must skip all prompts
    and silently apply Balanced defaults. Manifest contract: zero prompts,
    zero stderr interaction, exit 0, `profile = "balanced"` written to
    `~/.superlocalmemory/config.toml`.
    """
    home = tmp_path / "home"
    home.mkdir()
    # Pin benchmark signals so this test is independent of the host's live
    # free-RAM / cold-start / disk-free state. The point of this test is
    # TTY-detection + zero prompts + Balanced default, not benchmark logic.
    env = {
        "CI": "true",
        "SLM_INSTALL_FREE_RAM_MB": "8192",
        "SLM_INSTALL_COLD_START_MS": "150",
        "SLM_INSTALL_DISK_FREE_GB": "250",
    }
    result = _run(
        ["--dry-run", f"--home={home}", "--home-outside-home"],
        env=env,
    )
    assert result.returncode == 0, (
        f"expected exit 0 in non-TTY mode, got {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # Must not have emitted an interactive prompt (which would end in "? ").
    for line in result.stdout.splitlines():
        stripped = line.rstrip()
        assert not stripped.endswith("?"), (
            f"non-TTY path emitted an interactive prompt: {line!r}\n"
            f"full stdout:\n{result.stdout}"
        )
    # In --dry-run with ample resources, the installer defaults to Balanced.
    assert "balanced" in result.stdout.lower(), (
        f"expected 'balanced' profile selection in stdout:\n{result.stdout}"
    )


def test_benchmark_recommends_minimal_on_low_ram(tmp_path: Path) -> None:
    """The install-time benchmark simulates free-RAM / Python-cold-start /
    disk-free. With a synthetic low-RAM signal (env var SLM_INSTALL_FREE_RAM_MB
    below the Light threshold), the installer must recommend/downgrade to
    Minimal profile. Completes in <15s per MASTER-PLAN §5.1.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "CI": "true",  # non-TTY, auto-apply recommendation
        "SLM_INSTALL_FREE_RAM_MB": "400",  # below Light's 900 MB threshold
    }
    result = _run(
        ["--dry-run", f"--home={home}", "--home-outside-home"],
        env=env,
        timeout=20,
    )
    assert result.returncode == 0, (
        f"benchmark path failed: stdout={result.stdout} stderr={result.stderr}"
    )
    assert "minimal" in result.stdout.lower(), (
        f"low-RAM machine must get Minimal recommendation, got:\n{result.stdout}"
    )


def test_custom_profile_writes_config_toml(tmp_path: Path) -> None:
    """Custom mode with an explicit reply-file JSON must write
    `~/.superlocalmemory/config.toml` with the custom values. Uses
    --reply-file so the test doesn't need a real TTY.
    """
    home = tmp_path / "home"
    home.mkdir()
    reply_file = tmp_path / "replies.json"
    reply_file.write_text(json.dumps({
        "profile": "custom",
        "ram_ceiling_mb": 1500,
        "hot_path_hooks": "sync_async",
        "reranker": "onnx_int8_l6",
        "context_injection_tokens": 500,
        "skill_evolution_enabled": False,
        "evolution_llm": "haiku",
        "online_retrain_cadence": "50_outcomes",
        "consolidation_cadence": "6h_nightly",
    }))
    # No --dry-run: must actually write the file.
    result = _run(
        [
            f"--home={home}",
            "--home-outside-home",
            "--profile=custom",
            f"--reply-file={reply_file}",
        ],
    )
    assert result.returncode == 0, (
        f"custom profile write failed: stdout={result.stdout} stderr={result.stderr}"
    )
    cfg = home / ".superlocalmemory" / "config.toml"
    assert cfg.exists(), f"expected config.toml at {cfg}, stdout={result.stdout}"
    parsed = _read_toml_kv(cfg)
    # profile is a top-level scalar
    assert parsed.get("profile") == "custom", f"bad toml: {parsed}"
    # ram_ceiling_mb lives under [runtime]
    assert parsed.get("runtime", {}).get("ram_ceiling_mb") == 1500, (
        f"ram_ceiling_mb not persisted: {parsed}"
    )


def test_reconfigure_preserves_config_bak(tmp_path: Path) -> None:
    """If a previous config.toml exists and --reconfigure is passed, the
    installer must back it up to config.toml.bak before overwriting.
    Without --reconfigure it must refuse to overwrite (exit 0 with skip msg).
    """
    home = tmp_path / "home"
    slm_dir = home / ".superlocalmemory"
    slm_dir.mkdir(parents=True)
    prior = slm_dir / "config.toml"
    prior_content = '# prior user config\nprofile = "light"\n'
    prior.write_text(prior_content)

    # First: without --reconfigure, should NOT overwrite.
    result_skip = _run(
        [f"--home={home}", "--home-outside-home", "--profile=power"],
        env={"CI": "true"},
    )
    assert result_skip.returncode == 0, result_skip.stderr
    assert prior.read_text() == prior_content, (
        "installer must not overwrite existing config without --reconfigure"
    )

    # Second: with --reconfigure, should back up and overwrite.
    result_rc = _run(
        [f"--home={home}", "--home-outside-home", "--profile=power", "--reconfigure"],
        env={"CI": "true"},
    )
    assert result_rc.returncode == 0, result_rc.stderr
    bak = slm_dir / "config.toml.bak"
    assert bak.exists(), "config.toml.bak must exist after --reconfigure"
    assert bak.read_text() == prior_content, (
        "config.toml.bak must contain the prior config verbatim"
    )
    new_cfg = _read_toml_kv(prior)
    assert new_cfg.get("profile") == "power", (
        f"new config.toml must reflect --profile=power, got {new_cfg}"
    )


def test_skill_evolution_prompt_defaults_no(tmp_path: Path) -> None:
    """Manifest D3: skill_evolution opt-in default OFF. For Balanced/Power
    profiles the installer asks, but the default (pressing Enter / non-TTY)
    must be `false`.
    """
    home = tmp_path / "home"
    home.mkdir()
    # Non-TTY Balanced path — no prompts, skill_evolution must be OFF.
    result = _run(
        [f"--home={home}", "--home-outside-home", "--profile=balanced"],
        env={"CI": "true"},
    )
    assert result.returncode == 0, result.stderr
    cfg = home / ".superlocalmemory" / "config.toml"
    assert cfg.exists(), f"config not written: stdout={result.stdout}"
    parsed = _read_toml_kv(cfg)
    # Skill evolution lives under [evolution] section.
    evo = parsed.get("evolution", {})
    assert evo.get("enabled") is False, (
        f"skill evolution must default to false (opt-in), got {evo}"
    )


def test_no_opus_in_model_choices() -> None:
    """Hard gate: the LLM model choice list offered to the user must contain
    only Claude Haiku 4.5, Claude Sonnet 4.6, Ollama, and Skip. Opus must
    never appear. This test reads the installer source and asserts it
    contains no 'opus' string in any model-choice context, and that the
    expected choices are present.
    """
    assert SCRIPT.exists(), f"installer not found at {SCRIPT}"
    src = SCRIPT.read_text()
    low = src.lower()
    # Absolute ban — the stage5b gate also scans for this, but we enforce
    # here for a local, fast signal.
    assert "opus" not in low, (
        "No Opus in model choices — found 'opus' in installer source. "
        "Use claude-haiku-4-5 / claude-sonnet-4-6 / Ollama / Skip only."
    )
    # Required choices all present.
    assert "claude-haiku-4-5" in low, "missing claude-haiku-4-5 choice"
    assert "claude-sonnet-4-6" in low, "missing claude-sonnet-4-6 choice"
    assert "ollama" in low, "missing Ollama choice"
    # 'skip' is common; check we have a distinct Skip model choice.
    assert re.search(r"\bskip\b", low), "missing Skip model choice"
