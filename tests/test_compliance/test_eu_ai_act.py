# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for the EU AI Act deployment-posture report.

The operating mode can establish technical facts such as data locality. It
cannot determine an EU AI Act risk class or certify legal compliance without
the deployment's intended purpose and context.
"""

from __future__ import annotations

import pytest

from superlocalmemory.compliance.eu_ai_act import ComplianceReport, EUAIActChecker
from superlocalmemory.storage.models import Mode


@pytest.fixture()
def checker() -> EUAIActChecker:
    return EUAIActChecker()


@pytest.mark.parametrize("mode", list(Mode))
def test_mode_alone_never_certifies_legal_compliance(
    checker: EUAIActChecker,
    mode: Mode,
) -> None:
    report = checker.check_compliance(mode)
    assert report.compliant is None
    assert report.risk_category == "undetermined"
    assert report.transparency_met is None
    assert report.human_oversight is None
    assert report.deployment_context_required is True
    assert "intended purpose" in " ".join(report.findings).lower()


def test_mode_a_reports_only_verifiable_technical_posture(
    checker: EUAIActChecker,
) -> None:
    report = checker.check_compliance(Mode.A)
    assert report.data_stays_local is True
    assert report.uses_generative_ai is False


def test_mode_b_reports_local_generative_ai_posture(
    checker: EUAIActChecker,
) -> None:
    report = checker.check_compliance(Mode.B)
    assert report.data_stays_local is True
    assert report.uses_generative_ai is True


def test_mode_c_reports_cloud_processing_and_dpa_consideration(
    checker: EUAIActChecker,
) -> None:
    report = checker.check_compliance(Mode.C)
    assert report.data_stays_local is False
    assert report.uses_generative_ai is True
    assert "DPA" in " ".join(report.findings)


def test_verify_all_modes_returns_three_reports(checker: EUAIActChecker) -> None:
    reports = checker.verify_all_modes()
    assert set(reports) == {"a", "b", "c"}
    assert all(isinstance(report, ComplianceReport) for report in reports.values())
    assert all(report.compliant is None for report in reports.values())


def test_get_compliant_modes_cannot_certify_from_mode_alone(
    checker: EUAIActChecker,
) -> None:
    assert checker.get_compliant_modes() == []


def test_report_is_immutable(checker: EUAIActChecker) -> None:
    report = checker.check_compliance(Mode.A)
    with pytest.raises(AttributeError):
        report.compliant = True  # type: ignore[misc]
