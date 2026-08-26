"""
evaluator.py

Message quality evaluator for structured communication in CC4 MARL.

The evaluator determines how correct/useful a message was after the
environment produces the next state.

Flow:

    Sender
       |
       v
    StructuredMessage
       |
       v
    Receiver
       |
       v
    Environment step
       |
       v
    Ground-truth state
       |
       v
    MessageEvaluator
       |
       v
    message_quality [0, 1]
       |
       v
    DynamicTrust.update()

IMPORTANT
---------
The ground-truth state used here is training/evaluation-side information.
It must NOT be provided to the Blue agents as part of their observation.

This module does NOT:
    - modify trust
    - encode messages
    - decode messages
    - interact directly with CybORG
    - perform MAPPO updates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .schema import (
    EventType,
    HostStatus,
    Priority,
    StructuredMessage,
    TargetType,
    ThreatLevel,
)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class MessageEvaluation:
    """
    Result of evaluating one structured message.

    All scores are in [0, 1].

    Attributes
    ----------
    overall_score:
        Final message quality sent to the trust mechanism.

    event_score:
        Correctness of the reported event.

    target_score:
        Correctness of the reported target.

    threat_score:
        Correctness of the reported threat level.

    status_score:
        Correctness of the reported status.

    usefulness_score:
        Whether the information was operationally useful.

    confidence:
        Confidence declared by the sender.

    details:
        Human-readable debugging information.
    """

    overall_score: float

    event_score: float
    target_score: float
    threat_score: float
    status_score: float
    usefulness_score: float

    confidence: float

    details: Optional[Dict[str, Any]] = None

    def as_dict(self) -> dict:
        """Return the evaluation as a dictionary."""

        return {
            "overall_score": self.overall_score,
            "event_score": self.event_score,
            "target_score": self.target_score,
            "threat_score": self.threat_score,
            "status_score": self.status_score,
            "usefulness_score": self.usefulness_score,
            "confidence": self.confidence,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Message evaluator
# ---------------------------------------------------------------------------

class MessageEvaluator:
    """
    Evaluates structured cyber-security messages.

    The evaluator compares the sender's message against ground-truth
    information available to the training/evaluation process.

    Parameters
    ----------
    event_weight:
        Weight assigned to event correctness.

    target_weight:
        Weight assigned to target correctness.

    threat_weight:
        Weight assigned to threat-level correctness.

    status_weight:
        Weight assigned to status correctness.

    usefulness_weight:
        Weight assigned to operational usefulness.

    Notes
    -----
    The default weights sum to 1.0.

    The target receives a relatively high weight because a message
    identifying the wrong host/subnet is significantly less useful
    in a cyber-defense setting.
    """

    def __init__(
        self,
        event_weight: float = 0.25,
        target_weight: float = 0.30,
        threat_weight: float = 0.15,
        status_weight: float = 0.15,
        usefulness_weight: float = 0.15,
    ) -> None:

        weights = [
            event_weight,
            target_weight,
            threat_weight,
            status_weight,
            usefulness_weight,
        ]

        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                "Evaluation weights must be non-negative."
            )

        total = sum(weights)

        if total <= 0.0:
            raise ValueError(
                "At least one evaluation weight must be positive."
            )

        # Normalize automatically.
        self.event_weight = event_weight / total
        self.target_weight = target_weight / total
        self.threat_weight = threat_weight / total
        self.status_weight = status_weight / total
        self.usefulness_weight = usefulness_weight / total

    # ------------------------------------------------------------------
    # Public evaluation interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> MessageEvaluation:
        """
        Evaluate a structured message against pure, message-independent
        ground truth for the sender's zone.

        Parameters
        ----------
        message:
            Structured message sent by a Blue agent.

        ground_truth:
            Zone truth produced by CC4Env.get_ground_truth(sender_id).
            This is completely independent of `message` -- it describes
            what is actually true in the sender's zone, and it is THIS
            method's job (not the environment's) to grade the message
            against it. Expected shape:

                {
                    "target_status": {
                        target_id (int): {
                            "event_type":   EventType,
                            "threat_level": ThreatLevel,
                            "status":       HostStatus,
                        },
                        ...   # one entry per relevant host in the zone
                    },
                    "zone_quiet": bool,
                }

            Because the target the agent named is looked up directly in
            `target_status`, a claim about the WRONG host is never
            silently re-pointed at a different real host: the wrong host
            is simply absent from `target_status` and scores as incorrect.

        previous_state:
            Optional state before the environment step (unused when the
            zone truth carries an explicit usefulness signal, which it
            always does; kept for interface compatibility).

        current_state:
            Optional state after the environment step (see above).

        Returns
        -------
        MessageEvaluation
            Evaluation result containing a quality score in [0, 1].
        """

        if not isinstance(message, StructuredMessage):
            raise TypeError(
                "message must be a StructuredMessage."
            )

        if not isinstance(ground_truth, dict):
            raise TypeError(
                "ground_truth must be a dictionary."
            )

        target_status = ground_truth.get("target_status", {})

        if not isinstance(target_status, dict):
            target_status = {}

        zone_quiet = ground_truth.get(
            "zone_quiet",
            len(target_status) == 0,
        )

        # Decide -- message-independently -- whether the target the agent
        # named was correct. Tri-state:
        #   True  -> named a genuinely relevant host, or correctly reported
        #            a quiet zone
        #   False -> named a wrong/uninvolved host, or stayed silent while
        #            something was actually happening
        #   None  -> ungraded (SUBNET claim): host-level truth cannot verify
        #            a subnet-scoped claim, so the target is EXCLUDED from
        #            the score (its weight is renormalized away) rather than
        #            being assigned a neutral 0.5.
        target_correct = self._evaluate_target(
            message,
            target_status,
            zone_quiet,
        )

        # Reference fact (event_type / threat_level / status) that the
        # preserved event/threat/status sub-evaluators grade against. This
        # never lets the message rewrite the truth; it only selects WHICH
        # real host (or the quiet baseline) the claim is checked against.
        reference_fact = self._resolve_reference_fact(
            message,
            target_status,
            zone_quiet,
        )

        # Event / threat / status partial scoring is preserved unchanged and
        # is always graded -- even for a SUBNET claim, whose event/threat/
        # status can still be checked against the zone.
        event_score = self._evaluate_event(
            message,
            reference_fact,
        )

        threat_score = self._evaluate_threat(
            message,
            reference_fact,
        )

        status_score = self._evaluate_status(
            message,
            reference_fact,
        )

        # Target correctness maps to a score, or is EXCLUDED (None) when the
        # claim is ungraded (SUBNET).
        if target_correct is None:
            target_score = None
        else:
            target_score = 1.0 if target_correct else 0.0

        # Usefulness is DERIVED from target correctness -- there is no longer
        # a separate ground-truth "useful" flag. A correctly identified
        # relevant host / quiet zone is useful; a wrong target is not; an
        # ungraded target excludes usefulness too.
        if target_correct is None:
            usefulness_score = None
        else:
            usefulness_score = 1.0 if target_correct else 0.0

        # Weighted sum over ONLY the graded components. Any excluded
        # component (score is None) is dropped and its weight renormalized
        # away, so a SUBNET claim is scored over event + threat + status
        # rather than being handed a neutral target_score of 0.5.
        weighted_components = (
            (event_score, self.event_weight),
            (target_score, self.target_weight),
            (threat_score, self.threat_weight),
            (status_score, self.status_weight),
            (usefulness_score, self.usefulness_weight),
        )

        active_weight = sum(
            weight
            for score, weight in weighted_components
            if score is not None
        )

        if active_weight <= 0.0:
            # Defensive only: event/threat/status are always graded, so at
            # least three components are present in practice.
            overall_score = 0.5
        else:
            overall_score = (
                sum(
                    weight * score
                    for score, weight in weighted_components
                    if score is not None
                )
                / active_weight
            )

        overall_score = self._clamp(
            overall_score
        )

        excluded_components = [
            name
            for name, score in (
                ("target", target_score),
                ("usefulness", usefulness_score),
            )
            if score is None
        ]

        return MessageEvaluation(
            overall_score=overall_score,
            event_score=event_score,
            target_score=target_score,
            threat_score=threat_score,
            status_score=status_score,
            usefulness_score=usefulness_score,
            confidence=message.confidence,
            details={
                "message": message.as_dict(),
                "ground_truth": ground_truth,
                "graded": True,
                "reference_fact": reference_fact,
                "target_correct": target_correct,
                "excluded_components": excluded_components,
            },
        )

    # ------------------------------------------------------------------
    # Event evaluation
    # ------------------------------------------------------------------

    def _evaluate_event(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """
        Evaluate event-type correctness.

        Exact match gives 1.0.

        If the ground truth does not contain event information,
        the evaluator returns 0.5 rather than falsely declaring
        the message incorrect.
        """

        actual = ground_truth.get("event_type")

        if actual is None:
            return 0.5

        actual = self._normalize_enum(
            actual,
            EventType,
        )

        if actual is None:
            return 0.5

        if message.event_type == actual:
            return 1.0

        # NONE is particularly bad when an actual event exists.
        if message.event_type == EventType.NONE:
            return 0.0

        # Some cyber events are semantically related.
        related_events = {
            EventType.DISCOVERY: {
                EventType.SCAN,
                EventType.DISCOVERY,
            },
            EventType.SCAN: {
                EventType.SCAN,
                EventType.DISCOVERY,
            },
            EventType.SUSPICIOUS_ACTIVITY: {
                EventType.SUSPICIOUS_ACTIVITY,
                EventType.DISCOVERY,
                EventType.SCAN,
            },
            EventType.COMPROMISE: {
                EventType.COMPROMISE,
                EventType.PRIVILEGE_ESCALATION,
            },
            EventType.PRIVILEGE_ESCALATION: {
                EventType.PRIVILEGE_ESCALATION,
                EventType.COMPROMISE,
            },
            EventType.LATERAL_MOVEMENT: {
                EventType.LATERAL_MOVEMENT,
            },
            EventType.RECOVERY: {
                EventType.RECOVERY,
            },
        }

        if actual in related_events.get(
            message.event_type,
            set(),
        ):
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Target evaluation
    # ------------------------------------------------------------------

    def _evaluate_target(
        self,
        message: StructuredMessage,
        target_status: Dict[int, Dict[str, Any]],
        zone_quiet: bool,
    ) -> Optional[bool]:
        """
        Decide whether the target the message named was correct, purely by
        looking the claim up in the message-independent zone truth. The
        message never rewrites the truth: a wrong host is simply absent from
        ``target_status`` and is scored as incorrect -- it is never silently
        re-pointed at a different, genuinely-compromised host (the old
        wrong-target rescue bug).

            HOST + target_id present in target_status -> True
                (named a genuinely relevant host)

            HOST + target_id absent from target_status -> False
                (named a normal / uninvolved host: wrong target)

            NONE + zone_quiet -> True
                (correctly reported that nothing was happening)

            NONE + not zone_quiet -> False
                (stayed silent while something was actually happening)

            SUBNET -> None
                (ungraded: ground truth is host-level, so a subnet-scoped
                claim cannot be verified; the caller EXCLUDES it from the
                score and renormalizes the remaining weights)
        """

        if message.target_type == TargetType.HOST:
            return message.target_id in target_status

        if message.target_type == TargetType.SUBNET:
            return None

        # TargetType.NONE: no specific host was named. Correct precisely when
        # the zone really was quiet.
        return bool(zone_quiet)

    # ------------------------------------------------------------------
    # Message-independent target resolution
    # ------------------------------------------------------------------

    def _resolve_reference_fact(
        self,
        message: StructuredMessage,
        target_status: Dict[int, Dict[str, Any]],
        zone_quiet: bool,
    ) -> Dict[str, Any]:
        """
        Pick the reference fact (event_type / threat_level / status) that the
        preserved event/threat/status sub-evaluators grade the message
        against. This NEVER lets the message rewrite the truth -- it only
        selects WHICH real host (or the quiet baseline) the field-level
        claims are checked against:

            HOST present  -> that host's real fact.
            HOST absent   -> NORMAL / NONE baseline (uninvolved host).
            NONE / SUBNET, zone quiet     -> NORMAL / NONE baseline.
            NONE / SUBNET, zone not quiet -> the most severe real host in the
                                             zone, so a broad-but-directionally
                                             correct alert still earns partial
                                             event/threat/status credit.

        Whether the *target itself* was correct is decided separately in
        _evaluate_target(); this method only supplies the field-level truth.
        A host that is not genuinely relevant is absent from `target_status`,
        so a HOST claim about it is graded against the NORMAL / NONE fact --
        never re-pointed at a different, genuinely-compromised host.
        """

        normal_fact = {
            "event_type": EventType.NONE,
            "threat_level": ThreatLevel.LOW,
            "status": HostStatus.NORMAL,
        }

        # A specific host was named: grade against that exact host when it is
        # genuinely relevant, otherwise against the NORMAL / NONE baseline.
        if message.target_type == TargetType.HOST:

            fact = target_status.get(message.target_id)

            if fact is not None:
                return dict(fact)

            return dict(normal_fact)

        # No specific host (NONE) or a subnet-scoped claim (SUBNET): a quiet
        # zone grades against the NORMAL / NONE baseline; otherwise the
        # field-level claims are checked against the most severe real host.
        if zone_quiet or not target_status:
            return dict(normal_fact)

        most_severe = max(
            target_status.values(),
            key=lambda entry: int(entry["threat_level"]),
        )

        return dict(most_severe)

    # ------------------------------------------------------------------
    # Threat evaluation
    # ------------------------------------------------------------------

    def _evaluate_threat(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """
        Evaluate threat-level correctness.

        Exact level:
            1.0

        One level away:
            0.5

        Two or more levels away:
            0.0

        This is better than strict binary accuracy because predicting
        HIGH instead of CRITICAL is not equivalent to predicting LOW.
        """

        actual = ground_truth.get(
            "threat_level"
        )

        if actual is None:
            return 0.5

        actual = self._normalize_enum(
            actual,
            ThreatLevel,
        )

        if actual is None:
            return 0.5

        predicted = int(
            message.threat_level
        )

        actual = int(actual)

        difference = abs(
            predicted - actual
        )

        if difference == 0:
            return 1.0

        if difference == 1:
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Status evaluation
    # ------------------------------------------------------------------

    def _evaluate_status(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """Evaluate host/status correctness."""

        actual = ground_truth.get(
            "status"
        )

        # Support a simpler ground-truth representation.
        if actual is None:

            if ground_truth.get(
                "compromised"
            ) is True:
                actual = HostStatus.COMPROMISED

            elif ground_truth.get(
                "suspicious"
            ) is True:
                actual = HostStatus.SUSPICIOUS

            elif (
                "compromised" in ground_truth
                or "suspicious" in ground_truth
            ):
                actual = HostStatus.NORMAL

        if actual is None:
            return 0.5

        actual = self._normalize_enum(
            actual,
            HostStatus,
        )

        if actual is None:
            return 0.5

        if message.status == actual:
            return 1.0

        # Unknown is less informative than an actual state.
        if message.status == HostStatus.UNKNOWN:
            return 0.0

        # NORMAL vs SUSPICIOUS is partially related.
        if {
            message.status,
            actual,
        } == {
            HostStatus.NORMAL,
            HostStatus.SUSPICIOUS,
        }:
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def confidence_adjusted_score(
        self,
        evaluation: MessageEvaluation,
    ) -> float:
        """
        Produce a confidence-aware message quality score.

        The confidence is NOT used as proof of correctness.

        Instead, highly confident incorrect messages are penalized
        more strongly than uncertain incorrect messages.

        Correctness remains the dominant factor.
        """

        quality = evaluation.overall_score
        confidence = evaluation.confidence

        # Distance from a neutral confidence level.
        confidence_strength = abs(
            confidence - 0.5
        ) * 2.0

        if quality >= 0.5:
            # Correct information benefits slightly from confidence.
            adjusted = (
                quality
                + 0.10
                * confidence_strength
                * quality
            )
        else:
            # Incorrect high-confidence information is more damaging.
            penalty = (
                0.10
                * confidence_strength
                * (1.0 - quality)
            )

            adjusted = quality - penalty

        return self._clamp(
            adjusted
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """Clamp a scalar to [minimum, maximum]."""

        return max(
            minimum,
            min(maximum, float(value)),
        )

    @staticmethod
    def _normalize_enum(
        value: Any,
        enum_type,
    ):
        """
        Convert an arbitrary enum representation into the
        corresponding IntEnum.

        Supports:

            IntEnum
            integer
            enum name string
        """

        if isinstance(
            value,
            enum_type,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            try:
                return enum_type[
                    value.upper()
                ]
            except KeyError:
                return None

        try:
            return enum_type(
                int(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            return None