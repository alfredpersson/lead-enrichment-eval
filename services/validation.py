"""
Input validation: character caps and language scope.

Caps are enforced before any model call. Language detection runs via lingua;
inputs outside English/Swedish/German get a friendly out-of-scope response
without a model call.
"""

from __future__ import annotations

from dataclasses import dataclass

from lingua import Language, LanguageDetectorBuilder

PROFILE_CHAR_CAP = 4000
COMPANY_CHAR_CAP = 2000

SUPPORTED_LANGS = {"en", "sv", "de"}
LANG_TO_CODE = {
    Language.ENGLISH: "en",
    Language.SWEDISH: "sv",
    Language.GERMAN: "de",
}

_detector = (
    LanguageDetectorBuilder.from_languages(*LANG_TO_CODE.keys())
    .with_preloaded_language_models()
    .build()
)


@dataclass
class ValidationError(Exception):
    code: str
    message: str


def detect_lang(text: str) -> str:
    """Return ISO code (en/sv/de) or 'other' if outside the supported set."""
    detected = _detector.detect_language_of(text)
    if detected is None:
        return "other"
    return LANG_TO_CODE.get(detected, "other")


def check_input(profile: str, company: str | None) -> str:
    """Raise ValidationError if input is rejected. Return detected lang code."""
    if not profile or not profile.strip():
        raise ValidationError(code="empty_profile", message="Profile is required.")
    if len(profile) > PROFILE_CHAR_CAP:
        raise ValidationError(
            code="profile_too_long",
            message=f"Profile exceeds {PROFILE_CHAR_CAP:,} character cap.",
        )
    if company is not None and len(company) > COMPANY_CHAR_CAP:
        raise ValidationError(
            code="company_too_long",
            message=f"Company description exceeds {COMPANY_CHAR_CAP:,} character cap.",
        )
    combined = profile if not company else f"{profile}\n\n{company}"
    lang = detect_lang(combined)
    if lang not in SUPPORTED_LANGS:
        raise ValidationError(
            code="language_out_of_scope",
            message="The demo supports English, Swedish, and German inputs.",
        )
    return lang
