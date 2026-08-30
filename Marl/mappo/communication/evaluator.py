"""
evaluator.py (v2 -- minimal scope: single target_id, dual ground-truth maps)

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
    Ground-truth state (sender's zone -- receiver-independent)
       |
       v
    MessageEvaluator (correctness: receiver-independent
                       usefulness:  receiver-dependent, via
                                    receiver_relevance)
       |
       v
    message_quality [0, 1]  (per (sender, receiver) pair)
       |
       v
    DynamicTrust.update()

IMPORTANT
---------
The ground-truth state used here is training/evaluation-side information.
It must NOT be provided to the Blue agents as part of their observation.

HOST vs SUBNET targets -- separate vocabularies, both gradable
-------------------------------------------------------------
`target_id` is ONE field on StructuredMessage (the message format is not
changed by this fix -- see schema.py's 7-field MESSAGE_FIELDS), but its
MEANING depends on `target_type`: HOST indexes the host vocabulary,
SUBNET indexes a SEPARATE subnet vocabulary. Ground truth mirrors that
split with two independent maps -- `target_status` (HOST) and
`subnet_status` (SUBNET) -- expected from CC4Env.get_ground_truth(). A
HOST claim is looked up ONLY in `target_status`; a SUBNET claim is looked
up ONLY in `subnet_status`. Neither is ever consulted for the other
target_type, and a claim is never re-pointed at a different, real target
just because the one actually named was wrong -- an id absent from (or
out of range for) the relevant map scores as incorrect, full stop.

Only when a caller passes ground truth from an OLDER CC4Env that has no
"subnet_status" key AT ALL (as opposed to a present-but-empty dict) does
a SUBNET claim fall back to being excluded from the score entirely
(weights renormalized over the remaining components) rather than
fabricating a verdict it has no evidence for.

Receiver-specific trust
------------------------
Whether a message's claim about the SENDER's zone is factually correct
is an objective fact about the world -- it does not depend on who is
listening, so `event_score`, `target_score`, `threat_score`, and
`status_score` are computed exactly once per (sender, message) and are
never receiver-conditioned. Grading them against a *receiver's* zone
instead would be meaningless, since the message says nothing about the
receiver's zone.

Operational *usefulness*, however, genuinely is receiver-relative: the
same, objectively-correct report matters more to a receiver whose zone
is closely operationally linked to the sender's zone than to one whose
zone is not. `usefulness_score` therefore takes an optional
`receiver_relevance` in [0, 1] (supplied by the caller, which is where
CC4 topology knowledge belongs -- this module still does not know
anything about zones, subnets, or CybORG) and blends it in. Passing no
`receiver_relevance` (or 1.0) reproduces the previous, receiver-blind
behavior exactly, so existing callers are unaffected until they opt in.

This module does NOT:
    - decode messages
    - encode messages
    - generate messages
    - interact with CybORG
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
        Correctness of the reported target (HOST checked against
        target_status, SUBNET checked against subnet_status -- see
        module docstring). None when the claim could not be graded at
        all (only possible with ground truth from an older CC4Env
        lacking "subnet_status").

    threat_score:
        Correctness of the reported threat level.

    status_score:
        Correctness of the reported status.

    usefulness_score:
        Whether the information was operationally useful -- receiver-
        dependent (see module docstring), unlike the four scores above.
        None under the same ungraded condition as target_score.

    confidence:
        Confidence declared by the sender.

    details:
        Human-readable debugging information.
    """

    overall_score: float

    event_score: float
    target_score: Optional[float]
    threat_score: float
    status_score: float
    usefulness_score: Optional[float]

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
        receiver_relevance: float = 1.0,
    ) -> MessageEvaluation:
        """
        Evaluate a structured message against pure, message-independent
        ground truth for the sender's zone.

        Parameters
        ----------
        message:
            Structured message sent by a Blue agent. `target_id` is a
            single field (unchanged message format); its vocabulary is
            selected by `target_type` -- see module docstring.

        ground_truth:
            Zone truth produced by CC4Env.get_ground_truth(sender_id).
            This is completely independent of `message` -- it describes
            what is actually true in the sender's zone, and it is THIS
            method's job (not the environment's) to grade the message
            against it. Expected shape:

                {
                    "target_status": {
                        target_id (int, HOST vocabulary): {
                            "event_type":   EventType,
                            "threat_level": ThreatLevel,
                            "status":       HostStatus,
                        },
                        ...   # one entry per relevant host in the zone
                    },
                    "subnet_status": {
                        target_id (int, SUBNET vocabulary): {
                            "event_type":   EventType,
                            "threat_level": ThreatLevel,
                            "status":       HostStatus,
                        },
                        ...   # one entry per relevant subnet in the zone
                    },
                    "zone_quiet": bool,
                }

            "subnet_status" is optional for backward compatibility with
            an older CC4Env that only ever produced "target_status" --
            when it is entirely absent (not merely empty), a SUBNET
            claim is excluded from the score rather than graded. When
            it IS present, a SUBNET claim is graded exactly like a HOST
            claim: looked up directly in `subnet_status`, never
            re-pointed at a different target.

            Because the target the agent named is looked up directly in
            the relevant map, a claim about the WRONG host/subnet is
            never silently re-pointed at a different real one: the
            wrong (or out-of-range) target is simply absent and scores
            as incorrect.

            This is the SENDER's zone truth regardless of who receives
            the message -- correctness does not depend on the receiver.

        previous_state:
            Optional state before the environment step (unused when the
            zone truth carries an explicit usefulness signal, which it
            always does; kept for interface compatibility).

        current_state:
            Optional state after the environment step (see above).

        receiver_relevance:
            How operationally relevant the sender's zone is to the
            specific receiver this evaluation is being scored for, in
            [0, 1]. 1.0 (the default) means fully relevant and
            reproduces the previous receiver-blind behavior exactly, so
            existing callers that don't pass this are unaffected. This
            is the ONLY input that may legitimately vary per receiver
            for the same (sender, message) pair -- everything else
            graded here is an objective fact about the sender's zone.
            Callers (e.g. train.py) own the topology knowledge needed
            to compute this; this module intentionally does not.

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

        receiver_relevance = self._clamp(receiver_relevance)

        target_status = ground_truth.get("target_status", {})

        if not isinstance(target_status, dict):
            target_status = {}

        # Optional (backward-compat): None means an older CC4Env that
        # never produced this map at all, as opposed to a present-but-
        # empty map meaning "no relevant subnets right now". Only the
        # former falls back to leaving SUBNET claims ungraded.
        subnet_status = ground_truth.get("subnet_status")

        if subnet_status is not None and not isinstance(subnet_status, dict):
            subnet_status = {}

        zone_quiet = ground_truth.get(
            "zone_quiet",
            len(target_status) == 0,
        )

        # Decide -- message-independently -- whether the target the agent
        # named was correct. Tri-state:
        #   True  -> named a genuinely relevant host/subnet, or correctly
        #            reported a quiet zone
        #   False -> named a wrong, out-of-range, or uninvolved
        #            host/subnet, or stayed silent while something was
        #            actually happening
        #   None  -> ungraded: only possible for a SUBNET claim when the
        #            caller's ground truth has no "subnet_status" map at
        #            all (older CC4Env); the target is EXCLUDED from the
        #            score (its weight is renormalized away) rather than
        #            being assigned a neutral 0.5.
        #
        # TargetType.NONE never performs a target_id lookup at all (there
        # is no specific target to look up) -- it is graded purely on
        # whether the zone really was quiet, independent of target_id's
        # value.
        #
        # This is receiver-independent: it's a fact about the message and
        # the sender's zone, not about who's listening.
        target_correct = self._evaluate_target(
            message,
            target_status,
            subnet_status,
            zone_quiet,
        )

        # Reference fact (event_type / threat_level / status) that the
        # preserved event/threat/status sub-evaluators grade against. This
        # never lets the message rewrite the truth; it only selects WHICH
        # real host/subnet (or the quiet baseline) the claim is checked
        # against.
        reference_fact = self._resolve_reference_fact(
            message,
            target_status,
            subnet_status,
            zone_quiet,
        )

        # Event / threat / status partial scoring is preserved unchanged and
        # is always graded -- even when the target itself is ungraded, its
        # event/threat/status can still be checked against the zone. Also
        # receiver-independent, for the same reason as target_correct.
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
        # claim is ungraded (SUBNET with no subnet_status map available).
        # Receiver-independent.
        if target_correct is None:
            target_score = None
        else:
            target_score = 1.0 if target_correct else 0.0

        # Usefulness is the ONE receiver-dependent axis. Incorrect
        # information isn't useful to anyone, regardless of relevance.
        # Correct information's usefulness scales with how operationally
        # relevant the sender's zone is to THIS receiver: a 0.5 floor
        # keeps some general situational-awareness value even for a
        # low-relevance receiver, rising to 1.0 for a highly relevant one.
        # This is what lets alpha[sender, receiver]/beta[sender, receiver]
        # in trust.py -- already pairwise -- actually diverge across
        # receivers instead of every receiver getting an identical score.
        if target_correct is None:
            usefulness_score = None
        elif target_correct:
            usefulness_score = 0.5 + 0.5 * receiver_relevance
        else:
            usefulness_score = 0.0

        # Weighted sum over ONLY the graded components. Any excluded
        # component (score is None) is dropped and its weight renormalized
        # away, so an ungraded SUBNET claim is scored over event + threat +
        # status rather than being handed a neutral target_score of 0.5.
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
                "receiver_relevance": receiver_relevance,
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
        subnet_status: Optional[Dict[int, Dict[str, Any]]],
        zone_quiet: bool,
    ) -> Optional[bool]:
        """
        Decide whether the target the message named was correct, purely by
        looking `message.target_id` up in the message-independent map that
        `message.target_type` selects. The message never rewrites the
        truth: a wrong or out-of-range id is simply absent from the
        relevant map and is scored as incorrect -- it is never silently
        re-pointed at a different, genuinely-compromised target (the old
        wrong-target rescue bug), and it is never checked against the
        WRONG map (a SUBNET claim is never looked up in target_status, a
        HOST claim never in subnet_status).

            HOST + target_id present in target_status -> True
                (named a genuinely relevant host)

            HOST + target_id absent from target_status -> False
                (named a normal, uninvolved, or out-of-range host: wrong
                target)

            SUBNET, subnet_status available (not None)
                + target_id present in subnet_status -> True
                (named a genuinely relevant subnet)

            SUBNET, subnet_status available (not None)
                + target_id absent from subnet_status -> False
                (named a normal, uninvolved, or out-of-range subnet:
                wrong target)

            SUBNET, subnet_status is None (older CC4Env) -> None
                (ungraded: no subnet-level ground truth is available at
                all, so the caller EXCLUDES it from the score and
                renormalizes the remaining weights -- this is a
                fallback for backward compatibility only; a current
                CC4Env always supplies subnet_status)

            NONE -> True iff zone_quiet, else False
                (no target_id lookup is performed at all for a NONE
                claim -- it is graded purely on whether the zone really
                was quiet)
        """

        if message.target_type == TargetType.HOST:
            return message.target_id in target_status

        if message.target_type == TargetType.SUBNET:

            if subnet_status is None:
                # No subnet-level ground truth available at all --
                # leave ungraded rather than guessing.
                return None

            return message.target_id in subnet_status

        # TargetType.NONE: no specific host/subnet was named, so no
        # target_id lookup happens at all -- correct precisely when the
        # zone really was quiet.
        return bool(zone_quiet)

    # ------------------------------------------------------------------
    # Message-independent target resolution
    # ------------------------------------------------------------------

    def _resolve_reference_fact(
        self,
        message: StructuredMessage,
        target_status: Dict[int, Dict[str, Any]],
        subnet_status: Optional[Dict[int, Dict[str, Any]]],
        zone_quiet: bool,
    ) -> Dict[str, Any]:
        """
        Pick the reference fact (event_type / threat_level / status) that the
        preserved event/threat/status sub-evaluators grade the message
        against. This NEVER lets the message rewrite the truth -- it only
        selects WHICH real host/subnet (or the quiet baseline) the
        field-level claims are checked against:

            HOST present    -> that host's real fact (from target_status).
            HOST absent     -> NORMAL / NONE baseline (uninvolved host).

            SUBNET present (subnet_status available and target_id in it)
                            -> that subnet's real aggregate fact (from
                               subnet_status).
            SUBNET absent, subnet_status available
                            -> NORMAL / NONE baseline (uninvolved subnet).
            SUBNET, subnet_status is None (older CC4Env)
                            -> falls back to the zone-wide "most severe
                               real host" heuristic below, same as a
                               NONE claim, since no subnet-specific
                               truth exists to check against.

            NONE, zone quiet                -> NORMAL / NONE baseline.
            NONE, zone not quiet            -> the most severe real host
                                                in the zone (from
                                                target_status), so a
                                                broad-but-directionally-
                                                correct alert still earns
                                                partial event/threat/
                                                status credit.

        Whether the *target itself* was correct is decided separately in
        _evaluate_target(); this method only supplies the field-level truth.
        A host/subnet that is not genuinely relevant is absent from the
        relevant map, so a claim about it is graded against the NORMAL /
        NONE fact -- never re-pointed at a different, genuinely-compromised
        target, and never cross-checked against the WRONG map.
        """

        normal_fact = {
            "event_type": EventType.NONE,
            "threat_level": ThreatLevel.LOW,
            "status": HostStatus.NORMAL,
        }

        # A specific host was named: grade against that exact host when it is
        # genuinely relevant, otherwise against the NORMAL / NONE baseline.
        # Only ever consults target_status -- never subnet_status.
        if message.target_type == TargetType.HOST:

            fact = target_status.get(message.target_id)

            if fact is not None:
                return dict(fact)

            return dict(normal_fact)

        # A specific subnet was named, and real subnet-level ground truth
        # exists: grade against that exact subnet when it is genuinely
        # relevant, otherwise against the NORMAL / NONE baseline -- the
        # same treatment HOST claims get. Only ever consults
        # subnet_status -- never target_status.
        if message.target_type == TargetType.SUBNET and subnet_status is not None:

            fact = subnet_status.get(message.target_id)

            if fact is not None:
                return dict(fact)

            return dict(normal_fact)

        # No specific host (NONE), or a subnet-scoped claim with no
        # subnet-level ground truth available (older CC4Env): a quiet
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


# ============================================================================
# Basic validation / self-test
# ============================================================================

def _run_self_test() -> None:
    """
    Minimal smoke test covering both target types plus invalid ids and
    NONE. Run directly: `python -m Marl.mappo.communication.evaluator`.
    Not a substitute for a real test suite -- just enough to catch a
    regression of the wrong-target rescue bug or a HOST/SUBNET map
    cross-check mistake.
    """

    evaluator = MessageEvaluator()

    host_ground_truth = {
        "target_status": {
            3: {
                "event_type": EventType.COMPROMISE,
                "threat_level": ThreatLevel.HIGH,
                "status": HostStatus.COMPROMISED,
            },
        },
        "subnet_status": {
            1: {
                "event_type": EventType.SUSPICIOUS_ACTIVITY,
                "threat_level": ThreatLevel.MEDIUM,
                "status": HostStatus.SUSPICIOUS,
            },
        },
        "zone_quiet": False,
    }

    # 1. Correct HOST claim scores target_score == 1.0.
    correct_host_message = StructuredMessage(
        event_type=EventType.COMPROMISE,
        target_type=TargetType.HOST,
        target_id=3,
        threat_level=ThreatLevel.HIGH,
        confidence=0.9,
        status=HostStatus.COMPROMISED,
        priority=Priority.URGENT,
    )
    result = evaluator.evaluate(correct_host_message, host_ground_truth)
    assert result.target_score == 1.0, "Correct HOST claim should score target_score=1.0"

    # 2. Wrong HOST id must NOT be rescued onto the real compromised host,
    # and must NOT be checked against subnet_status either.
    wrong_host_message = StructuredMessage(
        event_type=EventType.COMPROMISE,
        target_type=TargetType.HOST,
        target_id=99,  # not in target_status
        threat_level=ThreatLevel.HIGH,
        confidence=0.9,
        status=HostStatus.COMPROMISED,
        priority=Priority.URGENT,
    )
    result = evaluator.evaluate(wrong_host_message, host_ground_truth)
    assert result.target_score == 0.0, "Wrong/out-of-range HOST id must score target_score=0.0"

    # 3. Correct SUBNET claim scores target_score == 1.0, graded against
    # subnet_status (never target_status).
    correct_subnet_message = StructuredMessage(
        event_type=EventType.SUSPICIOUS_ACTIVITY,
        target_type=TargetType.SUBNET,
        target_id=1,
        threat_level=ThreatLevel.MEDIUM,
        confidence=0.7,
        status=HostStatus.SUSPICIOUS,
        priority=Priority.HIGH,
    )
    result = evaluator.evaluate(correct_subnet_message, host_ground_truth)
    assert result.target_score == 1.0, "Correct SUBNET claim should score target_score=1.0"

    # 4. Wrong/out-of-range SUBNET id scores incorrect, not excluded (since
    # subnet_status IS present here).
    wrong_subnet_message = StructuredMessage(
        event_type=EventType.SUSPICIOUS_ACTIVITY,
        target_type=TargetType.SUBNET,
        target_id=42,  # not in subnet_status
        threat_level=ThreatLevel.MEDIUM,
        confidence=0.7,
        status=HostStatus.SUSPICIOUS,
        priority=Priority.HIGH,
    )
    result = evaluator.evaluate(wrong_subnet_message, host_ground_truth)
    assert result.target_score == 0.0, "Wrong/out-of-range SUBNET id must score target_score=0.0"

    # 5. SUBNET claim with NO subnet_status map at all (older CC4Env) is
    # excluded from scoring, not guessed at.
    legacy_ground_truth = {
        "target_status": host_ground_truth["target_status"],
        "zone_quiet": False,
        # no "subnet_status" key at all
    }
    result = evaluator.evaluate(correct_subnet_message, legacy_ground_truth)
    assert result.target_score is None, "SUBNET claim with no subnet_status must be excluded (None)"
    assert "target" in result.details["excluded_components"]

    # 6. NONE claim never performs a target_id lookup -- correctness
    # depends only on zone_quiet.
    none_message_while_active = StructuredMessage(
        event_type=EventType.NONE,
        target_type=TargetType.NONE,
        target_id=0,
        threat_level=ThreatLevel.LOW,
        confidence=0.5,
        status=HostStatus.NORMAL,
        priority=Priority.LOW,
    )
    result = evaluator.evaluate(none_message_while_active, host_ground_truth)
    assert result.target_score == 0.0, "NONE claim while zone is active must score target_score=0.0"

    quiet_ground_truth = {"target_status": {}, "subnet_status": {}, "zone_quiet": True}
    result = evaluator.evaluate(none_message_while_active, quiet_ground_truth)
    assert result.target_score == 1.0, "NONE claim while zone is quiet must score target_score=1.0"

    print("evaluator.py self-test passed: HOST and SUBNET claims are graded "
          "independently, invalid ids score incorrect (never rescued), and "
          "NONE is graded purely on zone_quiet.")


if __name__ == "__main__":
    _run_self_test()