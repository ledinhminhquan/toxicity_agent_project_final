# Problem Definition Document (1–2 pages)

## Business context & motivation
Online platforms (community forums, e-commerce reviews, internal enterprise collaboration tools) face:
- Increased moderation cost as user-generated content scales.
- Brand risk and user churn when harmful content is not handled quickly.
- Inconsistent enforcement when moderation relies only on humans.

This project builds a **Toxicity Detection & Moderation Agent** that:
1) detects toxic / hateful / threatening language,
2) recommends an action (allow / warn / block / human review),
3) logs signals for monitoring and continual improvement.

## Target users / stakeholders
- **Content moderators**: faster triage, fewer false negatives.
- **Trust & Safety**: policy enforcement analytics.
- **Product**: reduced user harm and improved retention.
- **Developers**: a deployable API for integration.

## Problem statement
Given a user comment, classify multiple toxicity types (multi-label) and decide an appropriate moderation action.

## Why NLP is required
Toxicity and hate speech are expressed in natural language with context, sarcasm, and ambiguity.
Rules/keywords alone are brittle and produce many false positives/negatives.

## Success metrics
### Business metrics
- Moderator time saved (minutes/comment)
- Reduction in harmful content exposure (e.g., % toxic comments blocked before publication)
- Reduction in escalations / user reports

### Technical metrics
- Multi-label F1 (micro/macro)
- ROC-AUC per label
- False negative rate for high-risk categories (e.g., threats)
- Latency: p50/p95 inference time per comment
