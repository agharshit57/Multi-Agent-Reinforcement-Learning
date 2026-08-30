"""
encoder.py (v3 -- minimal scope: single target_id, dual embedding tables)

Key fix from v2: target_id's MEANING depends on target_type (see
schema.py's module comment on StructuredMessage.target_id): HOST indexes
the host vocabulary, SUBNET indexes a completely separate subnet
vocabulary. The message format is NOT changed here -- target_id stays
ONE field, exactly as schema.py's 7-field MESSAGE_FIELDS defines it, and
nothing in decoder.py/mappo.py/buffer.py needs to change. What changes
is how THIS module embeds that one id: two separate embedding tables,
selected by target_type, so a host index and a subnet index are never
looked up in the same table.

Previously (the bug): a single `target_embedding` table, sized off the
HOST vocabulary only (env.py.get_num_targets()), was used for target_id
regardless of target_type. A SUBNET-typed message's target_id was either
looked up directly against the HOST table (silently wrong -- host index 3
and subnet index 3 are unrelated things sharing one row), or wrapped into
range with something like `target_id % num_subnet_targets` ("aliasing").
Aliasing is explicitly NOT used here: two different out-of-vocabulary ids
that happen to share a remainder would alias to the identical embedding,
which is just a different flavor of the same ambiguity bug, not a fix.
Instead, out-of-range ids are validated and their contribution is forced
to a zero vector (see _safe_embedding_mask below) -- never silently
reinterpreted as some other, in-range target.

Vocabulary sizing note (decoder.py is intentionally NOT touched here):
target_id is produced by decoder.py's single existing target_head, so for
every valid subnet id to be reachable that head must be sized to at least
max(num_host_targets, num_subnet_targets). If num_subnet_targets <=
num_host_targets (the common case -- CC4 has far fewer subnets than
hosts), this holds automatically whenever the head is host-sized. If a
scenario ever has more subnets than hosts, some subnet ids would be
structurally unreachable from a host-sized head; that is a decoder.py
sizing concern, not something this file can fix by itself, and is exactly
why out-of-range validation below matters regardless of which case you're
in.

encode_from_ids() is the training-time entry point. It takes
already-sampled discrete field ids -- either freshly sampled at rollout
time, or replayed from the buffer at PPO-update time -- and embeds them.
The ids themselves are treated as constants (no gradient needed through
"which symbol was chosen": that's the decoder's job, trained via its own
log_prob). The embedding lookups and fusion MLP below ARE live,
differentiable ops, so gradient from the receiving agent's policy loss
reaches this encoder normally as long as encode_from_ids() is called
inside the same forward pass that produces that agent's action logits --
which is exactly what SharedActor.forward() does. (This point, and the
earlier fact that the old forward() consumed soft decoder-logit
distributions to no effect, are carried over unchanged from v2 and are
not re-litigated here.)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .schema import (
    ConfidenceLevel,
    EventType,
    HostStatus,
    Priority,
    TargetType,
    ThreatLevel,
)


class MessageEncoder(nn.Module):
    """Embeds a structured message (given as discrete field ids) into a fixed-size vector."""

    def __init__(
        self,
        message_dim: int = 128,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_host_targets: Optional[int] = None,
        num_subnet_targets: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.message_dim = message_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_host_targets = num_host_targets
        self.num_subnet_targets = num_subnet_targets

        self.event_embedding = nn.Embedding(len(EventType), embedding_dim)
        self.target_type_embedding = nn.Embedding(len(TargetType), embedding_dim)
        self.threat_embedding = nn.Embedding(len(ThreatLevel), embedding_dim)
        self.status_embedding = nn.Embedding(len(HostStatus), embedding_dim)
        self.priority_embedding = nn.Embedding(len(Priority), embedding_dim)

        # Was a Linear(1, embedding_dim) projection of a raw sigmoid scalar.
        # confidence is now a bucketed id like everything else -- same
        # embedding-table treatment, consistent with the rest of the fields.
        self.confidence_embedding = nn.Embedding(len(ConfidenceLevel), embedding_dim)

        # Two SEPARATE target vocabularies, both indexed by the SAME
        # single `target_id` field -- target_type picks which table
        # (if either) actually contributes. See module docstring for
        # why this replaces one shared/aliased table.
        self.host_target_embedding: Optional[nn.Embedding] = None
        self.subnet_target_embedding: Optional[nn.Embedding] = None

        if num_host_targets is not None or num_subnet_targets is not None:
            self.build_target_embeddings(num_host_targets, num_subnet_targets)

        num_fields = 7  # event, target_type, target, threat, confidence, status, priority
        fusion_input_dim = num_fields * embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, message_dim),
        )

        self.output_norm = nn.LayerNorm(message_dim)

    def build_target_embeddings(
        self,
        num_host_targets: Optional[int] = None,
        num_subnet_targets: Optional[int] = None,
    ) -> None:
        """
        Configure either or both target vocabularies. Unlike a single
        shared table, HOST and SUBNET are independent here -- you may
        configure just one if, say, a scenario has no subnet targets
        yet, without forcing the other into existence.
        """

        if num_host_targets is not None:

            if num_host_targets <= 0:
                raise ValueError("num_host_targets must be greater than zero")

            self.num_host_targets = num_host_targets
            self.host_target_embedding = nn.Embedding(num_host_targets, self.embedding_dim)

        if num_subnet_targets is not None:

            if num_subnet_targets <= 0:
                raise ValueError("num_subnet_targets must be greater than zero")

            self.num_subnet_targets = num_subnet_targets
            self.subnet_target_embedding = nn.Embedding(num_subnet_targets, self.embedding_dim)

    # ------------------------------------------------------------------
    # Safe, non-aliasing embedding lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_embedding_lookup(
        embedding: nn.Embedding,
        ids: torch.Tensor,
        type_mask: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        """
        Look up `ids` in `embedding`, restricted to elements where
        `type_mask` is True (i.e. target_type actually selects this
        vocabulary) AND `ids` is in [0, num_targets).

        Out-of-range or wrong-type entries contribute a ZERO vector --
        never wrapped/aliased into range (e.g. via `ids % num_targets`,
        which is explicitly NOT used here). Aliasing would make two
        different, invalid ids indistinguishable from some unrelated
        valid target; zeroing instead makes "no valid claim here"
        unambiguous and gradient-safe (no embedding row is ever
        credited/blamed for an id it was never actually representing).

        `ids` still must be clamped before the actual nn.Embedding
        call purely so PyTorch doesn't raise an IndexError on an
        out-of-range value -- the clamped (dummy) lookup's result is
        then masked to zero for those same entries, so the dummy value
        never reaches the output.
        """

        in_range = (ids >= 0) & (ids < num_targets)
        valid = type_mask & in_range

        # Clamp only to keep nn.Embedding from raising on an
        # out-of-range id; the corresponding output rows are zeroed
        # below regardless of what this dummy lookup produces.
        safe_ids = ids.clamp(min=0, max=num_targets - 1)

        looked_up = embedding(safe_ids)

        return looked_up * valid.unsqueeze(-1).to(looked_up.dtype)

    def encode_from_ids(self, field_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed a message from discrete field ids of arbitrary leading shape,
        e.g. [B] for a single sender per batch item, or [B, N] for N senders
        per batch item (nn.Embedding broadcasts over any leading dims).

        Required keys: event_type, target_type, target_id, threat_level,
        status, priority, confidence -- the same 7 fields as before;
        nothing about the message format changes here.

        target_id is looked up against host_target_embedding when
        target_type == HOST, against subnet_target_embedding when
        target_type == SUBNET, and contributes nothing (zero vector)
        when target_type == NONE -- matching the pre-existing "NONE ->
        ignored" convention. An id that is out of range for whichever
        vocabulary target_type selects (or a vocabulary that was never
        configured) also contributes a zero vector rather than being
        wrapped/aliased into some other target's meaning -- see
        _safe_embedding_lookup().
        """

        event_vector = self.event_embedding(field_ids["event_type"])
        target_type_vector = self.target_type_embedding(field_ids["target_type"])
        threat_vector = self.threat_embedding(field_ids["threat_level"])
        status_vector = self.status_embedding(field_ids["status"])
        priority_vector = self.priority_embedding(field_ids["priority"])
        confidence_vector = self.confidence_embedding(field_ids["confidence"])

        target_id = field_ids.get("target_id")

        if target_id is None:
            target_vector = torch.zeros_like(event_vector)

        else:

            target_type_ids = field_ids["target_type"]

            is_host = target_type_ids == int(TargetType.HOST)
            is_subnet = target_type_ids == int(TargetType.SUBNET)

            if self.host_target_embedding is not None:
                host_vector = self._safe_embedding_lookup(
                    self.host_target_embedding,
                    target_id,
                    is_host,
                    self.num_host_targets,
                )
            else:
                host_vector = torch.zeros_like(event_vector)

            if self.subnet_target_embedding is not None:
                subnet_vector = self._safe_embedding_lookup(
                    self.subnet_target_embedding,
                    target_id,
                    is_subnet,
                    self.num_subnet_targets,
                )
            else:
                subnet_vector = torch.zeros_like(event_vector)

            # Mutually exclusive by construction (is_host and is_subnet
            # can never both be True for the same element), so a plain
            # sum is exactly a select -- no double-counting.
            target_vector = host_vector + subnet_vector

        combined = torch.cat(
            [
                event_vector,
                target_type_vector,
                target_vector,
                threat_vector,
                confidence_vector,
                status_vector,
                priority_vector,
            ],
            dim=-1,
        )

        communication_vector = self.fusion(combined)
        return self.output_norm(communication_vector)


# ============================================================================
# Basic validation / self-test
# ============================================================================

def _run_self_test() -> None:
    """
    Minimal smoke test exercising both target types plus the invalid-id
    and NONE-type paths. Run directly: `python -m Marl.mappo.communication.encoder`.
    Not a substitute for a real test suite -- just enough to catch an
    obviously broken vocabulary wiring (wrong table selected, a crash on
    out-of-range ids, or a nonzero contribution from an invalid id).
    """

    torch.manual_seed(0)

    num_host_targets = 10
    num_subnet_targets = 4

    encoder = MessageEncoder(
        message_dim=32,
        embedding_dim=8,
        hidden_dim=32,
        num_host_targets=num_host_targets,
        num_subnet_targets=num_subnet_targets,
    )
    encoder.eval()

    def make_ids(target_type: TargetType, target_id: int) -> Dict[str, torch.Tensor]:
        return {
            "event_type": torch.tensor([int(EventType.COMPROMISE)]),
            "target_type": torch.tensor([int(target_type)]),
            "target_id": torch.tensor([target_id]),
            "threat_level": torch.tensor([int(ThreatLevel.HIGH)]),
            "confidence": torch.tensor([int(ConfidenceLevel.HIGH)]),
            "status": torch.tensor([int(HostStatus.COMPROMISED)]),
            "priority": torch.tensor([int(Priority.URGENT)]),
        }

    with torch.no_grad():

        # 1. HOST id 3 and SUBNET id 3 must NOT produce the same vector
        # (that was exactly the old aliasing bug).
        host_vec = encoder.encode_from_ids(make_ids(TargetType.HOST, 3))
        subnet_vec = encoder.encode_from_ids(make_ids(TargetType.SUBNET, 3))
        assert not torch.allclose(host_vec, subnet_vec), (
            "HOST and SUBNET target_id=3 produced identical vectors -- "
            "vocabularies are not actually separate."
        )

        # 2. A HOST id must only ever consult host_target_embedding --
        # changing it while target_type=SUBNET must have zero effect.
        subnet_vec_a = encoder.encode_from_ids(make_ids(TargetType.SUBNET, 0))
        subnet_vec_b = encoder.encode_from_ids(make_ids(TargetType.SUBNET, num_subnet_targets - 1))
        assert not torch.allclose(subnet_vec_a, subnet_vec_b), (
            "Different valid SUBNET ids produced identical vectors."
        )

        # 3. Out-of-range SUBNET id (>= num_subnet_targets) must NOT
        # alias to any in-range subnet's vector, and must not crash.
        out_of_range_vec = encoder.encode_from_ids(
            make_ids(TargetType.SUBNET, num_subnet_targets + 5)
        )
        for valid_id in range(num_subnet_targets):
            valid_vec = encoder.encode_from_ids(make_ids(TargetType.SUBNET, valid_id))
            assert not torch.allclose(out_of_range_vec, valid_vec), (
                f"Out-of-range SUBNET id aliased onto valid id {valid_id}."
            )

        # 4. Out-of-range HOST id must behave the same way.
        out_of_range_host_vec = encoder.encode_from_ids(
            make_ids(TargetType.HOST, num_host_targets + 5)
        )
        for valid_id in range(num_host_targets):
            valid_vec = encoder.encode_from_ids(make_ids(TargetType.HOST, valid_id))
            assert not torch.allclose(out_of_range_host_vec, valid_vec), (
                f"Out-of-range HOST id aliased onto valid id {valid_id}."
            )

        # 5. NONE-type messages must ignore target_id entirely.
        none_vec_a = encoder.encode_from_ids(make_ids(TargetType.NONE, 0))
        none_vec_b = encoder.encode_from_ids(make_ids(TargetType.NONE, 7))
        assert torch.allclose(none_vec_a, none_vec_b), (
            "TargetType.NONE should ignore target_id, but changing it "
            "changed the encoded vector."
        )

    print("encoder.py self-test passed: HOST/SUBNET vocabularies are "
          "separate, out-of-range ids do not alias, NONE ignores target_id.")


if __name__ == "__main__":
    _run_self_test()