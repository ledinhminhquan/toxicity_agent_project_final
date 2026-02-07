from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..utils import redact_pii, sha256_text
from ..models.model_registry import Predictors
from .actions import Action, ModerationDecision
from .human_review_queue import HumanReviewQueue
from .language import LanguageDetector, is_english
from .policy_store import PolicyStore


@dataclass
class ModerationAgent:
    predictors: Predictors
    policy: PolicyStore
    lang_detector: LanguageDetector
    review_queue: Optional[HumanReviewQueue]
    cfg: Dict

    def moderate(self, text: str) -> ModerationDecision:
        raw_text = text or ""
        lang = self.lang_detector.detect(raw_text)

        # -------- Step 1: run fast model --------
        fast_probs = self.predictors.detoxify_fast.predict_proba_matrix(
            [raw_text],
            label_order=self.predictors.label_fields,
            label_map=self.predictors.label_map_detoxify,
        )[0]
        fast_scores = {lf: float(fast_probs[i]) for i, lf in enumerate(self.predictors.label_fields)}
        fast_overall = float(np.max(fast_probs))

        agent_cfg = self.cfg.get("agent", {})
        borderline_low = float(agent_cfg.get("borderline_low", 0.35))
        borderline_high = float(agent_cfg.get("borderline_high", 0.65))
        high_risk_threshold = float(agent_cfg.get("high_risk_threshold", 0.85))

        borderline = borderline_low <= fast_overall <= borderline_high
        high_risk = fast_overall >= high_risk_threshold

        # -------- Step 2: routing decision (agentic) --------
        model_used = f"detoxify:{self.predictors.detoxify_fast.model_type}"
        final_scores = dict(fast_scores)
        final_overall = fast_overall

        prefer_multi = bool(agent_cfg.get("prefer_multilingual_when_not_english", True))

        need_second_pass = borderline or high_risk or (lang == "unknown")

        if need_second_pass:
            if (not is_english(lang)) and prefer_multi:
                # Use multilingual model for non-English
                multi_probs = self.predictors.detoxify_multilingual.predict_proba_matrix(
                    [raw_text],
                    label_order=self.predictors.label_fields,
                    label_map=self.predictors.label_map_detoxify,
                )[0]
                final_scores = {lf: float(multi_probs[i]) for i, lf in enumerate(self.predictors.label_fields)}
                final_overall = float(np.max(multi_probs))
                model_used = f"detoxify:{self.predictors.detoxify_multilingual.model_type}"
            elif is_english(lang) and self.predictors.finetuned is not None:
                # Use fine-tuned model for English when available
                hf_probs = self.predictors.finetuned.predict_proba_matrix(
                    [raw_text],
                    label_order=self.predictors.label_fields,
                )[0]
                final_scores = {lf: float(hf_probs[i]) for i, lf in enumerate(self.predictors.label_fields)}
                final_overall = float(np.max(hf_probs))
                model_used = f"finetuned:{self.predictors.finetuned.model_dir.name}"
            # else: stick with fast model

        # -------- Step 3: policy decision --------
        thresholds = agent_cfg.get("action_thresholds", {})
        allow_lt = float(thresholds.get("allow_lt", 0.30))
        warn_lt = float(thresholds.get("warn_lt", 0.60))
        block_gte = float(thresholds.get("block_gte", 0.60))

        if final_overall < allow_lt:
            action = Action.ALLOW
        elif final_overall < warn_lt:
            action = Action.WARN
        else:
            action = Action.BLOCK

        # REVIEW overrides for uncertainty
        request_review = False
        if lang == "unknown":
            action = Action.REVIEW
            request_review = True
        elif borderline and action != Action.ALLOW:
            # Borderline case: prefer human review instead of auto block
            action = Action.REVIEW
            request_review = True

        # -------- Step 4: explanation --------
        top_k = self.policy.explanation_top_k()
        include_scores = self.policy.explanation_include_scores()

        sorted_labels = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        parts = []
        for label, score in sorted_labels:
            name = self.policy.label_display_name(label)
            if include_scores:
                parts.append(f"{name}: {score:.2f}")
            else:
                parts.append(name)
        explanation = "Top signals -> " + ", ".join(parts) if parts else "No strong toxicity signals."

        policy_message = self.policy.action_message(action)

        # -------- Step 5: enqueue for human review (tool usage) --------
        redacted = redact_pii(raw_text)
        if request_review and self.review_queue is not None:
            self.review_queue.enqueue(
                {
                    "text_sha256": sha256_text(raw_text),
                    "language": lang,
                    "overall_score": final_overall,
                    "label_scores": final_scores,
                    "model_used": model_used,
                }
            )

        return ModerationDecision(
            action=action,
            overall_score=final_overall,
            label_scores=final_scores,
            model_used=model_used,
            language=lang,
            policy_message=policy_message,
            explanation=explanation,
            request_human_review=request_review,
            redacted_for_logs=redacted,
        )
