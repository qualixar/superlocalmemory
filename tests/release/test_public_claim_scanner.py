from pathlib import Path

from scripts.claim_scanner import scan_paths


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scanner_rejects_unqualified_claims(tmp_path: Path) -> None:
    public_file = tmp_path / "README.md"
    public_file.write_text(
        "3 peer-reviewed papers. 1M entries. Zero slowdown. "
        "IDE integrations: 17+. Smart compression up to 32x. 74.8% on LoCoMo.\n",
        encoding="utf-8",
    )

    rule_ids = {finding.rule.rule_id for finding in scan_paths([public_file])}

    assert {
        "peer-review-status",
        "unproven-scale",
        "embedding-compression-as-product",
        "integration-count",
        "unqualified-locomo-74",
    } <= rule_ids


def test_scanner_accepts_properly_scoped_historical_results(tmp_path: Path) -> None:
    public_file = tmp_path / "README.md"
    public_file.write_text(
        "Historical V3 research result: 74.8% used local retrieval with "
        "GPT-4.1-mini answer construction. Historical Mode C result: "
        "87.7% on 81 questions from one conversation. These are not V3.7 results.\n",
        encoding="utf-8",
    )

    assert scan_paths([public_file]) == []


def test_scanner_rejects_stale_license_in_runtime_source(tmp_path: Path) -> None:
    runtime_file = tmp_path / "src" / "superlocalmemory" / "runtime.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text(
        "# SPDX-License-Identifier: " + "Elastic-" + "2.0\n", encoding="utf-8"
    )

    rule_ids = {finding.rule.rule_id for finding in scan_paths([tmp_path])}

    assert "license-inconsistency" in rule_ids


def test_scanner_allows_stale_license_only_in_historical_archive(tmp_path: Path) -> None:
    archive = tmp_path / "docs" / "v2-archive" / "README.md"
    archive.parent.mkdir(parents=True)
    archive.write_text(
        "Historical license: " + "Elastic License " + "2.0\n", encoding="utf-8"
    )

    assert scan_paths([tmp_path]) == []


def test_scanner_checks_rendered_html_in_dist_directory(tmp_path: Path) -> None:
    rendered = tmp_path / "dist" / "index.html"
    rendered.parent.mkdir()
    rendered.write_text("<h1>The world's first AI agent memory</h1>\n", encoding="utf-8")

    rule_ids = {finding.rule.rule_id for finding in scan_paths([rendered.parent])}

    assert "unsupported-superlative" in rule_ids


def test_scanner_rejects_mit_on_current_product_license_surfaces(tmp_path: Path) -> None:
    citation = tmp_path / "CITATION.cff"
    citation.write_text("license: MIT\n", encoding="utf-8")
    signer = tmp_path / "src" / "superlocalmemory" / "attribution" / "signer.py"
    signer.parent.mkdir(parents=True)
    signer.write_text('_LICENSE: str = "MIT"\n', encoding="utf-8")

    findings = scan_paths([tmp_path])

    assert {finding.path.name for finding in findings} >= {"CITATION.cff", "signer.py"}
    assert {finding.rule.rule_id for finding in findings} == {"license-inconsistency"}


def test_current_repository_public_claims_pass_scanner() -> None:
    findings = scan_paths([REPO_ROOT])
    details = "\n".join(
        f"{finding.path.relative_to(REPO_ROOT)}:{finding.line}: {finding.rule.rule_id}"
        for finding in findings
    )
    assert findings == [], details
