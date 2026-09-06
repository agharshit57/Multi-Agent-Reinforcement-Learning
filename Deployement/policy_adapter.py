"""Policy-compatibility layer: REAL topology -> frozen CC4 contracts.

This is the ONLY module allowed to mention real assets and CC4 slots in
the same breath. Everything else sees exactly one side:

* ``real_world.py`` / collectors .... real assets, ``asset_id`` keys.
* ``normalizer/state_builder/observation/action_table/action_mask/
  policy_engine`` .................... CC4 slots, EXACT training bytes.
* ``real_actions.py`` ................. policy decisions -> real targets.

Mapping rules (deterministic, audited, no pretending):

1. Real assets sorted by ``asset_id`` fill policy slots in
   ``AssetMap.stable_order()`` (global training order). First N win;
   the rest are ``overflow`` (inventory only, never fed to the model).
2. Slots with no physical asset are ``unpopulated``: they carry explicit
   EMPTY telemetry (quiet under ANY baselines -- verified against
   ``normalizer._fold_host``) and are labelled
   ``unpopulated-policy-slot-*`` on TEST-NET-3 (RFC 5737, never real).
   They are a technical padding of the fixed 112-slot policy geometry,
   NOT healthy machines, and surfaces must display them only inside a
   "policy mapping (technical)" drawer -- see ``MappingReport``.
3. Bound slots carry the REAL asset's ``ip``/``hostname``/``role`` in
   the generated asset map, so live backends resolve real targets
   through the UNCHANGED ``executor`` path. Unpopulated slots can never
   resolve to anything real (see the phantom-target guard in
   ``real_actions.py``).
4. Bound-but-silent real assets (no telemetry this batch) are OMITTED
   from the adapted batch: the frozen normalizer reads absence as
   STALE (fail-safe alert posture). Silence is never quiet.

Frozen contracts (never altered here, only satisfied): 112 slots,
137 vocab, 92/210 obs, 82/242 actions, slot order, mask semantics.
"""

import hashlib

from .asset_map import AGENT_ZONES, AssetMap, match_zone
from .telemetry import HostTelemetry, TelemetryBatch

# RFC 5737 TEST-NET-3: documentation-only, never routes, never real.
UNPOPULATED_IP_PREFIX = "203.0.113."
UNPOPULATED_HOSTNAME_PREFIX = "unpopulated-policy-slot"
POLICY_SOURCE_TAG = "policy-adapter"


class PolicyAdapterError(ValueError):
    """Real topology cannot be represented in the policy geometry."""


class MappingReport:
    """Audit record of one real->policy binding (JSON-serializable)."""

    def __init__(self, bindings, unpopulated, overflow):
        # bindings: [{asset_id, hostname, ips, agent, cc4,
        #             slot_ip, slot_hostname}]
        self.bindings = list(bindings)
        self.unpopulated = list(unpopulated)   # [cc4...]
        self.overflow = list(overflow)         # [asset_id...]
        self.digest = _digest(self.bindings)

    def asset_of_slot(self, cc4):
        for binding in self.bindings:
            if binding["cc4"] == cc4:
                return binding["asset_id"]
        return None

    def slot_of_asset(self, asset_id):
        for binding in self.bindings:
            if binding["asset_id"] == asset_id:
                return (binding["agent"], binding["cc4"])
        return None

    def bound_assets_in_policy_zone(self, policy_zone):
        """Real asset_ids bound to slots of one CC4 policy zone."""
        out = []
        for binding in self.bindings:
            try:
                if match_zone(binding["cc4"]) == policy_zone:
                    out.append(binding["asset_id"])
            except Exception:
                continue
        return sorted(out)

    def to_dict(self):
        return {"bindings": [dict(b) for b in self.bindings],
                "unpopulated": list(self.unpopulated),
                "overflow": list(self.overflow),
                "digest": self.digest}

    def summary(self):
        return (f"policy mapping {self.digest}: {len(self.bindings)} "
                f"real asset(s) bound, {len(self.unpopulated)} "
                f"unpopulated policy slots, {len(self.overflow)} "
                f"overflow (inventory only)")


def _digest(bindings):
    canonical = "\n".join(
        f"{b['asset_id']}|{b['agent']}|{b['cc4']}"
        for b in sorted(bindings,
                        key=lambda b: (b["agent"], b["cc4"])))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _owning_agent(cc4):
    """Agent whose zones contain this slot (zones are unique per agent)."""
    zone = match_zone(cc4)
    owners = [agent for agent, zones in AGENT_ZONES.items()
              if zone in zones]
    if len(owners) != 1:
        raise PolicyAdapterError(
            f"policy slot {cc4!r} has {len(owners)} owning agents "
            f"(zone {zone!r}); policy geometry ambiguous")
    return f"blue_agent_{owners[0]}"


class PolicyAdapter:
    """Binds a real topology into a CC4 policy map + adapts batches."""

    def __init__(self, real_topology):
        self.topology = real_topology

    # ------------------------------------------------- map building --
    def build_policy_map(self, previous=None):
        """(AssetMap, MappingReport) from the real topology.

        The returned AssetMap satisfies the frozen 112-slot contract and
        is consumable by the UNCHANGED pipeline; the report says what
        is real, what is padding, and what did not fit.

        Bindings are STICKY: ``previous`` (an earlier MappingReport for
        the same site) keeps its slots for assets that are still
        present, so a stable fleet never shifts policy slots between
        cycles. New assets (sorted) take the lowest free slots;
        departed assets free theirs; overflow is sorted. A first build
        (``previous=None``) is positional: sorted assets fill
        ``stable_order()`` slots in order -- fully deterministic.
        """
        assets = sorted(self.topology.assets,
                        key=lambda a: a.asset_id)
        slots = AssetMap.stable_order()
        if len(slots) != 112:
            raise PolicyAdapterError(
                f"policy geometry drift: {len(slots)} slots != 112")
        present = {a.asset_id: a for a in assets}
        # Sticky carry-over: previous slots survive for assets that are
        # still present (same slot set => same digest => stable policy
        # view across cycles; changes are audited via the new report).
        sticky, free = {}, list(slots)
        if previous is not None:
            for binding in previous.bindings:
                asset_id, cc4 = (binding["asset_id"], binding["cc4"])
                if asset_id in present and cc4 in free:
                    sticky[asset_id] = cc4
                    free.remove(cc4)
        bindings, overflow = [], []
        per_agent = {}
        placeholder_n = [0]

        def placeholder():
            placeholder_n[0] += 1
            n = placeholder_n[0]
            if n > 250:
                raise PolicyAdapterError(
                    "too many unpopulated policy slots for TEST-NET-3")
            return (f"{UNPOPULATED_IP_PREFIX}{n}",
                    f"{UNPOPULATED_HOSTNAME_PREFIX}-{n:03d}")

        slot_of = dict(sticky)
        newcomers = sorted(a for a in present if a not in slot_of)
        if len(newcomers) > len(free):
            overflow_ids = newcomers[len(free):]
            newcomers = newcomers[:len(free)]
        else:
            overflow_ids = []
        for asset_id in newcomers:
            slot_of[asset_id] = free.pop(0)
        for position, cc4 in enumerate(slots):
            agent = _owning_agent(cc4)
            asset_id = next((aid for aid, slot in slot_of.items()
                             if slot == cc4), None)
            if asset_id is not None:
                asset = present[asset_id]
                ip = _primary_real_ip(asset)
                slot_hostname = asset.hostname or asset.asset_id
                entry = {"cc4": cc4, "ip": ip,
                         "hostname": slot_hostname, "role": ""}
                bindings.append({
                    "asset_id": asset.asset_id,
                    "hostname": asset.hostname,
                    "ips": list(asset.ips or ()),
                    "agent": agent, "cc4": cc4,
                    # Coverage travels with the binding so displays can
                    # separate sensed assets from uncovered/observed ones.
                    "coverage": asset.coverage or "unknown",
                    # Telemetry key the normalizer resolves (ip
                    # preferred, hostname fallback -- mirrors the
                    # frozen reverse index; never invented).
                    "slot_ip": ip or slot_hostname,
                    "slot_hostname": slot_hostname})
            else:
                slot_ip, slot_hostname = placeholder()
                entry = {"cc4": cc4, "ip": slot_ip,
                         "hostname": slot_hostname, "role": ""}
            per_agent.setdefault(agent, []).append(entry)
        overflow = list(overflow_ids)
        agents = {f"blue_agent_{i}": {"hosts": per_agent.get(
            f"blue_agent_{i}", [])} for i in range(5)}
        # Overflow assets stay visible as inventory (dicts pass the
        # asset-map 'unmapped must be objects' check; they are never
        # slots and never reach the model).
        unmapped = [{"asset_id": present[aid].asset_id,
                     "hostname": present[aid].hostname,
                     "ips": list(present[aid].ips or ()),
                     "note": "overflow: no free policy slot"}
                    for aid in overflow_ids]
        policy_map = AssetMap.from_dict(
            {"agents": agents, "unmapped": unmapped})
        unpopulated = [cc4 for cc4 in slots
                       if _report_unpopulated(cc4, bindings)]
        report = MappingReport(bindings, unpopulated, overflow)
        return policy_map, report

    # ------------------------------------------------ batch adapting --
    def adapt_batch(self, real_batch, policy_map, report):
        """Translate a real (asset_id-keyed) batch to a slot-keyed one.

        Bound slots get verbatim real telemetry under the slot key;
        unpopulated slots get explicit EMPTY (quiet) telemetry;
        bound-but-silent assets are omitted (frozen STALE semantics);
        unknown keys are ignored and counted (never fed, never quiet).
        """
        slot_hosts = {}
        by_asset = {b["asset_id"]: b for b in report.bindings}
        silent, ignored = [], []
        for asset_id, binding in by_asset.items():
            tele = real_batch.hosts.get(asset_id)
            if tele is None:
                silent.append(asset_id)
                continue
            slot_hosts[binding["slot_ip"]] = HostTelemetry(
                key=binding["slot_ip"],
                processes=list(tele.processes or []),
                connections=list(tele.connections or []),
                sessions=list(tele.sessions or []),
                up=bool(tele.up),
                events=list(tele.events or []))
        for cc4 in report.unpopulated:
            key = _slot_ip(policy_map, cc4)
            slot_hosts[key] = HostTelemetry(key=key, processes=[],
                                            connections=[], sessions=[],
                                            up=True, events=[])
        for key in real_batch.hosts:
            if key not in by_asset:
                ignored.append(key)
        notes = (f"{POLICY_SOURCE_TAG} from {real_batch.source or 'real'}: "
                 f"{len(by_asset) - len(silent)} bound emitting, "
                 f"{len(silent)} bound silent (stale), "
                 f"{len(report.unpopulated)} unpopulated (quiet padding), "
                 f"{len(ignored)} ignored, "
                 f"{len(report.overflow)} overflow (inventory)")
        if silent:
            notes += f"; silent: {','.join(sorted(silent))}"
        if ignored:
            notes += f"; ignored keys: {','.join(sorted(ignored))}"
        return TelemetryBatch(
            timestamp=float(real_batch.timestamp),
            hosts=slot_hosts, notes=notes,
            source=(f"{POLICY_SOURCE_TAG}({real_batch.source})"
                    if real_batch.source else POLICY_SOURCE_TAG),
            partial_errors=list(real_batch.partial_errors or []))


def _primary_real_ip(asset):
    for ip in asset.ips or ():
        text = str(ip)
        if text and not text.startswith("127.") and text != "::1":
            return text
    if asset.ips:
        return str(asset.ips[0])
    # No observed address: fall back to the stable asset_id as the
    # hostname-keyed slot (AssetMap allows empty ip; the normalizer
    # then resolves via hostname). Never invent an address.
    return ""


def _slot_ip(policy_map, cc4):
    for agent in policy_map.agents:
        info = policy_map.agents[agent]["hosts"].get(cc4)
        if info is not None:
            return info.get("ip", "") or info.get("hostname", "")
    raise PolicyAdapterError(f"policy slot {cc4!r} missing from map")


def _report_unpopulated(cc4, bindings):
    return all(b["cc4"] != cc4 for b in bindings)


__all__ = [
    "PolicyAdapter", "PolicyAdapterError", "MappingReport",
    "UNPOPULATED_IP_PREFIX", "UNPOPULATED_HOSTNAME_PREFIX",
    "POLICY_SOURCE_TAG",
]
