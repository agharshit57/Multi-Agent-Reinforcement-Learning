"""CC4 state builder: normalized telemetry -> simulation-shaped state.

Produces the single snapshot the observation builder consumes:
  - mission_phase (configured, default Preplanning/0)
  - blocks: {to_zone: [from_zone, ...]} from enforcement state
  - per-host flags: compromised / process_event / connection_event

This mirrors what ``BlueFlatWrapper.observation_change`` reads from
``state`` (mission phase, ``state.blocks``, per-host event checks),
without importing any simulator code.
"""

from dataclasses import dataclass, field


@dataclass
class CC4State:
    mission_phase: int = 0
    blocks: dict = field(default_factory=dict)   # to_zone -> [from_zone]
    hosts: dict = field(default_factory=dict)    # cc4 -> NormalizedHost


class CC4StateBuilder:
    def __init__(self, mission_phase=0):
        self.mission_phase = int(mission_phase)

    def build(self, normalized, enforcement):
        """Combine telemetry signals with current enforcement state."""
        blocks = {}
        for to_zone, from_zones in (enforcement.get("blocks") or {}).items():
            active = sorted(set(from_zones))
            if active:
                blocks[to_zone] = active
        return CC4State(mission_phase=self.mission_phase,
                        blocks=blocks,
                        hosts=dict(normalized.hosts))

    def set_mission_phase(self, phase):
        phase = int(phase)
        if phase not in (0, 1, 2):
            raise ValueError(f"mission_phase must be 0/1/2, got {phase}")
        self.mission_phase = phase
