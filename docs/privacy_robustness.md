# Data Privacy & Robustness

## Privacy risks
User text can contain PII:
- emails, phone numbers, addresses, names, account identifiers.

Mitigations:
- Default logging stores only SHA-256 hash of the text.
- Optional redaction utilities for emails/phones/URLs.
- Avoid printing raw samples in notebooks or logs.
- Access control for any stored raw text (if enabled).

## Security
- No hard-coded credentials.
- Secrets should be provided via environment variables or secret managers.

## Robustness considerations
- Out-of-domain: new slang, code-switching, domain-specific jargon.
- Adversarial: obfuscation (leetspeak), inserted punctuation, unicode tricks.

Mitigations:
- Normalize whitespace and unicode.
- Add character-level augmentation (future work).
- Human review for borderline cases.
- Monitor drift and retrain when needed.
