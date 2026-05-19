"""Tests for input validation: char caps and language scope."""

from __future__ import annotations

import pytest

from services.validation import (
    COMPANY_CHAR_CAP,
    PROFILE_CHAR_CAP,
    SUPPORTED_LANGS,
    ValidationError,
    check_input,
    detect_lang,
)


def test_detect_lang_english():
    assert detect_lang("Hello, I am a software engineer building B2B SaaS tools.") == "en"


def test_detect_lang_swedish():
    assert detect_lang(
        "Hej, jag är en mjukvaruutvecklare som bygger verktyg för företag."
    ) == "sv"


def test_detect_lang_german():
    assert detect_lang(
        "Hallo, ich bin ein Softwareentwickler und baue Werkzeuge für Unternehmen."
    ) == "de"


def test_detect_lang_other_returns_other():
    # Japanese — outside supported set.
    assert detect_lang("こんにちは、私はソフトウェアエンジニアです。日本語を話します。") == "other"


def test_supported_langs_contract():
    # Lock the canonical set so dropping a language is a deliberate change.
    assert SUPPORTED_LANGS == {"en", "sv", "de"}


def test_check_input_happy_english():
    assert check_input("Maya Chen is VP Product at Lattice Forge.", None) == "en"


def test_check_input_happy_with_company():
    lang = check_input(
        "Maya Chen is VP Product.",
        "Lattice Forge is a Series B B2B SaaS company.",
    )
    assert lang == "en"


def test_check_input_empty_profile_raises():
    with pytest.raises(ValidationError) as exc:
        check_input("", None)
    assert exc.value.code == "empty_profile"


def test_check_input_whitespace_only_profile_raises():
    with pytest.raises(ValidationError) as exc:
        check_input("   \n\t   ", None)
    assert exc.value.code == "empty_profile"


def test_check_input_profile_over_cap_raises():
    over_cap = "x" * (PROFILE_CHAR_CAP + 1)
    with pytest.raises(ValidationError) as exc:
        check_input(over_cap, None)
    assert exc.value.code == "profile_too_long"
    assert str(PROFILE_CHAR_CAP) in exc.value.message.replace(",", "")


def test_check_input_profile_at_cap_passes():
    # English content padded to cap. The detector needs enough English signal
    # to classify, so seed with a real sentence then pad with spaces.
    seed = "Maya Chen is VP Product at Lattice Forge, a B2B SaaS company. "
    profile = seed + " " * (PROFILE_CHAR_CAP - len(seed))
    assert len(profile) == PROFILE_CHAR_CAP
    assert check_input(profile, None) == "en"


def test_check_input_company_over_cap_raises():
    over_cap = "x" * (COMPANY_CHAR_CAP + 1)
    with pytest.raises(ValidationError) as exc:
        check_input("Maya Chen is VP Product.", over_cap)
    assert exc.value.code == "company_too_long"


def test_check_input_language_out_of_scope_raises():
    # Japanese profile.
    profile = "こんにちは、私はソフトウェアエンジニアです。日本語を話します。"
    with pytest.raises(ValidationError) as exc:
        check_input(profile, None)
    assert exc.value.code == "language_out_of_scope"
