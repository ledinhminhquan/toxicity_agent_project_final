# Ethics Impact Statement

## Who benefits
- Users in communities: reduced exposure to harmful content.
- Moderators: reduced workload and improved triage.
- Platforms: improved trust and safety outcomes.

## Who could be harmed
- Users whose content is incorrectly flagged (false positives).
- Vulnerable groups if the model exhibits identity-term bias.

## Bias & fairness risks
Toxicity detectors often over-predict toxicity for text mentioning certain identities.
We mitigate by:
- Using Detoxify "unbiased" baseline.
- Requiring human review for borderline cases.
- Proposing fairness slice evaluations (e.g., identity mention groups).

## Explainability for stakeholders
We provide:
- top contributing label probabilities (not token-level explanations),
- clear action rationale,
- audit logs for moderation decisions (privacy-preserving).

## Misuse risks
- Over-reliance on automation; mitigate with human-in-the-loop.
- Using the model to target/harass users; avoid exposing raw scores broadly.

This system is intended to assist moderation, not replace human judgment.
