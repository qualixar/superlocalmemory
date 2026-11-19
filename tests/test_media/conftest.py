import pytest


@pytest.fixture(autouse=True)
def _flags_the_old_default_assumed_off(monkeypatch):
    """``media.regions`` and ``trust.secret_screen`` are on by default now. Regions make the reader return page
    boxes (the fake readers in these tests return none) and the screen strips a key from saved words. These
    tests are about the plain read and verbatim words, so both are set off explicitly; the tests of regions and
    of the screen turn their own flag on."""
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "off")
    monkeypatch.setenv("SLM_FEATURE_TRUST_SECRET_SCREEN", "off")
