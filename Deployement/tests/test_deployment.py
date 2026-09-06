"""Deployment-layer tests (stdlib + numpy only; no torch/CybORG/Marl).

Run from the repo root:
    python -m unittest Deployement.tests.test_deployment -v
Covers exact training-geometry replication: observation dims/layout,
action table counts and anchor indices, mask safety net, validator
modes, asset-map validation, and a headless mock-policy pipeline run.
"""

import os
import sys
import tempfile
import unittest

import numpy as np


def _telemetry_hosts(asset_map, **overrides):
    from Deployement.telemetry import HostTelemetry
    hosts = {}
    for agent in asset_map.agents.values():
        for info in agent["hosts"].values():
            hosts[info["ip"]] = HostTelemetry(key=info["ip"])
    hosts.update(overrides)
    return hosts


class TestConstants(unittest.TestCase):
    def test_stable_host_universe(self):
        from Deployement import config
        self.assertEqual(config.NUM_HOST_TARGETS, 137)
        self.assertEqual(list(config.STABLE_HOST_LIST),
                         sorted(config.STABLE_HOST_LIST))
        # Anchor: lexicographic position is deterministic by construction.
        idx = config.STABLE_HOST_INDEX[
            "restricted_zone_a_subnet_server_host_0"]
        self.assertEqual(config.STABLE_HOST_LIST[idx],
                         "restricted_zone_a_subnet_server_host_0")

    def test_dims(self):
        from Deployement import config
        self.assertEqual((config.SMALL_OBS_DIM, config.LARGE_OBS_DIM,
                          config.OBS_DIM), (92, 210, 210))
        self.assertEqual((config.SMALL_ACTION_DIM, config.LARGE_ACTION_DIM,
                          config.ACTION_DIM), (82, 242, 242))
        self.assertEqual(config.SUBNET_BLOCK_DIM, 59)
        self.assertEqual(config.MESSAGE_DIM, 32)

    def test_communication_field_order(self):
        # Buffer/decoder field order the comm path is built against.
        from Deployement import config
        self.assertEqual(config.MESSAGE_FIELD_NAMES,
                         ("event_type", "target_type", "threat_level",
                          "status", "priority", "confidence", "target_id"))


class TestAssetMap(unittest.TestCase):
    def test_default_map_validates(self):
        from Deployement.asset_map import generate_default_map
        amap = generate_default_map()
        total = sum(len(a["hosts"]) for a in amap.agents.values())
        self.assertEqual(total, 4 * 16 + 3 * 16)  # 112 slots, HQ has 3 zones

    def test_slot_order_matches_training(self):
        from Deployement.asset_map import (generate_default_map,
                                           zone_slot_hostnames)
        amap = generate_default_map()
        self.assertEqual(
            amap.slot_of("blue_agent_0",
                         "restricted_zone_a_subnet_server_host_0"), (0, 0))
        self.assertEqual(
            amap.slot_of("blue_agent_0",
                         "restricted_zone_a_subnet_user_host_0"), (0, 6))
        self.assertEqual(zone_slot_hostnames(
            "restricted_zone_a_subnet")[:2],
            ["restricted_zone_a_subnet_server_host_0",
             "restricted_zone_a_subnet_server_host_1"])

    def test_rejects_missing_slot(self):
        from Deployement.asset_map import AssetMapError, generate_default_map
        amap = generate_default_map()
        data = amap.to_dict()
        hosts = data["agents"]["blue_agent_0"]["hosts"]
        data["agents"]["blue_agent_0"]["hosts"] = hosts[1:]
        from Deployement.asset_map import AssetMap
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_roundtrip(self):
        from Deployement.asset_map import AssetMap, generate_default_map
        amap = generate_default_map()
        amap2 = AssetMap.from_dict(amap.to_dict())
        self.assertEqual(amap.to_dict(), amap2.to_dict())


class TestObservation(unittest.TestCase):
    def _state(self):
        from Deployement.asset_map import generate_default_map
        from Deployement.normalizer import NormalizedHost, NormalizedState
        from Deployement.state_builder import CC4State
        amap = generate_default_map()
        hosts = {}
        for agent in amap.agents.values():
            for cc4 in agent["hosts"]:
                hosts[cc4] = NormalizedHost(cc4=cc4, key="x")
        return amap, CC4State(mission_phase=0, blocks={}, hosts=hosts)

    def test_batch_dims(self):
        from Deployement.observation import ObservationBuilder, assert_layout
        assert_layout()
        amap, state = self._state()
        batch = ObservationBuilder(amap).build_batch(state)
        self.assertEqual(batch.shape, (5, 210))
        self.assertEqual(batch.dtype, np.float32)

    def test_native_dims_and_tail(self):
        from Deployement.observation import ObservationBuilder
        amap, state = self._state()
        builder = ObservationBuilder(amap)
        native0 = builder.build_native(0, state)
        self.assertEqual(native0.shape, (92,))
        self.assertTrue((native0[-32:] == 0).all())  # native msg block
        padded0 = builder.build_padded(0, state)
        self.assertEqual(padded0.shape, (210,))
        self.assertTrue((padded0[-32:] == 0).all())  # tail position
        # Mission + one-hot of the agent's own zone in slot 0.
        self.assertEqual(padded0[0], 0)
        one_hot = padded0[1:10]
        self.assertEqual(one_hot.sum(), 1)
        self.assertEqual(int(np.argmax(one_hot)), 7)  # restricted_zone_a
        native4 = builder.build_native(4, state)
        self.assertEqual(native4.shape, (210,))

    def test_alerts_land_in_slots(self):
        from Deployement.observation import ObservationBuilder
        amap, state = self._state()
        state.hosts["restricted_zone_a_subnet_server_host_0"].process_event = True  # noqa: E501
        state.hosts["restricted_zone_a_subnet_user_host_3"].connection_event = True  # noqa: E501
        vec = ObservationBuilder(amap).build_padded(0, state)
        proc = vec[1 + 27:1 + 27 + 16]
        conn = vec[1 + 27 + 16:1 + 27 + 32]
        self.assertEqual(proc[0], 1)   # server_host_0 -> slot 0
        self.assertEqual(proc[1:].sum(), 0)
        self.assertEqual(conn[9], 1)   # user_host_3 -> slot 6+3=9
        self.assertEqual(conn.sum(), 1)

    def test_blocks_and_phase(self):
        from Deployement.observation import ObservationBuilder
        amap, state = self._state()
        state.blocks = {"restricted_zone_a_subnet":
                        ["contractor_network_subnet"]}
        state.mission_phase = 1
        vec = ObservationBuilder(amap).build_padded(0, state)
        blocked = vec[1 + 9:1 + 18]
        self.assertEqual(blocked[1], 1)  # contractor index in SUBNETS
        self.assertEqual(blocked.sum(), 1)
        self.assertEqual(vec[0], 1)


class TestActionTable(unittest.TestCase):
    def test_counts(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        tables = build_all_tables(generate_default_map())
        for agent in range(4):
            self.assertEqual(len(tables[agent]), 82)
        self.assertEqual(len(tables[4]), 242)

    def test_anchor_indices_agent0(self):
        from Deployement.action_table import build_action_table
        from Deployement.asset_map import generate_default_map
        table = build_action_table(0, generate_default_map())
        self.assertEqual(table[0]["command"], "Analyse")
        self.assertEqual(table[0]["target"],
                         "restricted_zone_a_subnet_server_host_0")
        self.assertEqual(table[16]["command"], "Monitor")
        self.assertEqual(table[17]["command"], "Remove")
        self.assertEqual(table[33]["command"], "Restore")
        self.assertEqual(table[49]["command"], "Sleep")
        self.assertEqual(table[50]["command"], "AllowTrafficZone")
        self.assertEqual(table[50]["target"],
                         "admin_network_subnet")  # first src alphabetically
        self.assertEqual(table[58]["command"], "BlockTrafficZone")
        self.assertEqual(table[66]["command"], "DeployDecoy")
        self.assertEqual(table[81]["target"],
                         "restricted_zone_a_subnet_user_host_9")

    def test_anchor_indices_hq(self):
        from Deployement.action_table import build_action_table
        from Deployement.asset_map import generate_default_map
        table = build_action_table(4, generate_default_map())
        self.assertEqual(table[0]["target"],
                         "admin_network_subnet_server_host_0")
        self.assertEqual(table[48]["command"], "Monitor")
        self.assertEqual(table[145]["command"], "Sleep")
        self.assertEqual(table[146]["command"], "AllowTrafficZone")
        self.assertEqual(table[170]["command"], "BlockTrafficZone")
        self.assertEqual(table[194]["command"], "DeployDecoy")
        self.assertEqual(len(table), 242)

    def test_assets_bound(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        tables = build_all_tables(generate_default_map())
        for agent, table in tables.items():
            for entry in table:
                if entry["kind"] == "host":
                    self.assertTrue(entry["asset"]["ip"])


class TestMask(unittest.TestCase):
    def test_disruptive_gating_and_safety_net(self):
        from Deployement.action_mask import compute_mask
        from Deployement.action_table import build_action_table
        from Deployement.asset_map import generate_default_map
        from Deployement.normalizer import NormalizedHost, NormalizedState
        amap = generate_default_map()
        table = build_action_table(0, amap)
        hosts = {cc4: NormalizedHost(cc4=cc4, key="x")
                 for cc4 in amap.agents["blue_agent_0"]["hosts"]}
        state = NormalizedState(hosts=hosts, unseen_keys=[])
        mask = compute_mask(table, state,
                            ("restricted_zone_a_subnet",))
        # Nothing flagged: Restore/Remove suppressed. Block stays
        # allowed: every source is outside this agent's visible zone
        # (unknown sources are left to the structural mask, mirroring
        # the training fix).
        for i, entry in enumerate(table):
            if entry["command"] in ("Restore", "Remove"):
                self.assertFalse(mask[i], entry)
            if entry["command"] in ("Sleep", "Monitor", "Analyse",
                                    "DeployDecoy", "BlockTrafficZone",
                                    "AllowTrafficZone"):
                self.assertTrue(mask[i], entry)
        self.assertTrue(mask.any())  # safety net
        # HQ agent sees admin/office/public: a Block whose source is a
        # visible-but-quiet zone IS suppressed.
        from Deployement.action_table import build_action_table as _bat
        hq_table = _bat(4, amap)
        hq_hosts = {cc4: NormalizedHost(cc4=cc4, key="x")
                    for cc4 in amap.agents["blue_agent_4"]["hosts"]}
        hq_mask = compute_mask(
            hq_table, NormalizedState(hosts=hq_hosts, unseen_keys=[]),
            ("admin_network_subnet", "office_network_subnet",
             "public_access_zone_subnet"))
        admin_src = [i for i, e in enumerate(hq_table)
                     if e["command"] == "BlockTrafficZone"
                     and e["target"] == "admin_network_subnet"]
        self.assertTrue(admin_src)
        for i in admin_src:
            self.assertFalse(hq_mask[i])
        contractor_src = [i for i, e in enumerate(hq_table)
                          if e["command"] == "BlockTrafficZone"
                          and e["target"] == "contractor_network_subnet"]
        for i in contractor_src:
            self.assertTrue(hq_mask[i])  # unseen source stays allowed
        # Flag one host: its Restore/Remove open up.
        hosts["restricted_zone_a_subnet_server_host_0"].process_event = True
        mask2 = compute_mask(table, state,
                             ("restricted_zone_a_subnet",))
        restore_idx = next(i for i, e in enumerate(table)
                           if e["command"] == "Restore"
                           and e["target"] ==
                           "restricted_zone_a_subnet_server_host_0")
        self.assertTrue(mask2[restore_idx])

    def test_pad(self):
        from Deployement.action_mask import pad_mask
        import numpy as np
        padded = pad_mask(np.ones(82, dtype=bool), 242)
        self.assertEqual(padded.shape, (242,))
        self.assertTrue(padded[:82].all())
        self.assertFalse(padded[82:].any())


class TestValidator(unittest.TestCase):
    def _setup(self, mode):
        from Deployement.action_table import build_action_table
        from Deployement.asset_map import generate_default_map
        from Deployement.validator import ActionValidator
        amap = generate_default_map()
        table = build_action_table(0, amap)
        mask = np.ones(len(table), dtype=bool)
        return ActionValidator(mode=mode, cooldown_s=0), table, mask

    def test_shadow_never_enforces(self):
        validator, table, mask = self._setup("shadow")
        restore = next(i for i, e in enumerate(table)
                       if e["command"] == "Restore")
        decision = validator.validate(0, restore, table, mask)
        self.assertFalse(decision["enforce"])
        self.assertFalse(decision["approved"])
        self.assertEqual(decision["operation"], "reimage_host")

    def test_masked_action_rejected(self):
        validator, table, mask = self._setup("shadow")
        mask[:] = False
        mask[49] = True  # Sleep only
        with self.assertRaises(Exception):
            validator.validate(0, 0, table, mask)

    def test_supervised_queues_destructive(self):
        validator, table, mask = self._setup("supervised")
        restore = next(i for i, e in enumerate(table)
                       if e["command"] == "Restore")
        queued = validator.validate(0, restore, table, mask)
        self.assertTrue(queued["needs_approval"])
        self.assertFalse(queued["approved"])
        approved = validator.approve(queued)
        self.assertTrue(approved["approved"])
        monitor = next(i for i, e in enumerate(table)
                       if e["command"] == "Monitor")
        auto = validator.validate(0, monitor, table, mask)
        self.assertTrue(auto["approved"])
        self.assertFalse(auto["needs_approval"])


class TestPipeline(unittest.TestCase):
    def test_headless_demo_run(self):
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        from Deployement.action_table import build_all_tables
        amap = generate_default_map()
        tables = build_all_tables(amap)
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap), MockPolicyEngine(tables),
            mode="shadow", baselines=demo_baselines())
        records = pipeline.run()
        pipeline.close()
        # 5 cycles x 5 agents.
        self.assertEqual(len(records), 25)
        for record in records:
            self.assertIn(record["status"], ("logged",))
            self.assertIn(record["command"],
                          ("Analyse", "Monitor", "Remove", "Restore", "Sleep",
                           "AllowTrafficZone", "BlockTrafficZone",
                           "DeployDecoy"))
        # Compromise cycle must surface a flagged host somewhere.
        flagged = [r for r in records if r["cycle"] == 2]
        self.assertTrue(flagged)

    def test_mock_mode_applies(self):
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        from Deployement.action_table import build_all_tables
        amap = generate_default_map()
        tables = build_all_tables(amap)
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap), MockPolicyEngine(tables),
            mode="mock", baselines=demo_baselines())
        pipeline.run(max_cycles=3)
        pipeline.close()
        self.assertGreaterEqual(pipeline.cycle, 3)

    def test_model_loader_missing_checkpoint(self):
        from Deployement.model_loader import DeploymentError, load_trained_mappo
        with self.assertRaises(DeploymentError):
            load_trained_mappo("checkpoints/does-not-exist.pt")

    def test_valid_mask_stable_positions(self):
        # Guards the subset-ordering trap: mask bits must sit at
        # STABLE_HOST_LIST positions (decoder head order).
        import os
        from Deployement import config
        from Deployement.asset_map import AssetMap
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        from Deployement.action_table import build_all_tables
        example = os.path.join("Deployement", "assets.example.json")
        amap = AssetMap.from_json_file(example)
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)),
            mode="shadow", baselines=demo_baselines())
        from Deployement.normalizer import NormalizedHost, NormalizedState
        hosts = {c: NormalizedHost(cc4=c, key="x")
                 for a in amap.agents.values() for c in a["hosts"]}
        masks, _, valid = pipeline._masks(
            NormalizedState(hosts=hosts, unseen_keys=[]))
        router_pos = config.STABLE_HOST_INDEX[
            "restricted_zone_a_subnet_router"]
        srv_pos = config.STABLE_HOST_INDEX[
            "restricted_zone_a_subnet_server_host_0"]
        op_pos = config.STABLE_HOST_INDEX[
            "operational_zone_a_subnet_server_host_0"]
        self.assertTrue(valid[0][srv_pos])     # own-zone host
        self.assertFalse(valid[0][router_pos])  # routers have no slots
        self.assertFalse(valid[0][op_pos])      # out-of-zone host
        pipeline.close()

    def test_example_assets_load(self):
        import os
        from Deployement.asset_map import AssetMap
        amap = AssetMap.from_json_file(
            os.path.join("Deployement", "assets.example.json"))
        self.assertEqual(len(amap.agents), 5)

    def test_gui_imports_without_display(self):
        import importlib
        module = importlib.import_module("Deployement.gui")
        self.assertTrue(hasattr(module, "launch"))
        self.assertTrue(hasattr(module, "DeploymentConsole"))


class TestGuiPresentation(unittest.TestCase):
    def test_mode_styles_cover_all_modes(self):
        from Deployement import gui
        from Deployement.config import MODES
        seen_labels = set()
        for mode in MODES:
            style = gui.mode_style(mode)
            for key in ("bg", "fg", "label", "blurb"):
                self.assertIn(key, style, mode)
                self.assertTrue(style[key], f"{mode}.{key}")
            seen_labels.add(style["label"])
        # Every mode banner is visually distinct.
        self.assertEqual(len(seen_labels), len(MODES))

    def test_mode_style_distinguishes_risk(self):
        from Deployement import gui
        self.assertNotEqual(gui.mode_style("shadow")["bg"],
                            gui.mode_style("live")["bg"])
        self.assertIn("REAL", gui.mode_style("live")["blurb"])

    def test_unknown_mode_never_crashes_banner(self):
        from Deployement import gui
        style = gui.mode_style("hyperdrive")
        self.assertTrue(style["bg"] and style["label"])

    def test_quick_start_guidance_present(self):
        from Deployement import gui
        text = gui.QUICK_START_TEXT
        for step in ("Rebuild", "Start", "Step", "Approvals"):
            self.assertIn(step, text)


class TestGuiLive(unittest.TestCase):
    """Widget-level tests (need a display; skipped headless)."""

    def _root(self):
        try:
            import tkinter
        except Exception:
            self.skipTest("tkinter unavailable")
        try:
            root = tkinter.Tk()
            root.withdraw()
            self.addCleanup(root.destroy)
            return root
        except Exception as exc:
            self.skipTest(f"no display: {exc}")

    def _console(self, root):
        from Deployement import gui
        return gui.DeploymentConsole(
            root,
            lambda: (_ for _ in ()).throw(
                AssertionError("pipeline must not build yet")))

    def test_console_constructs_banner_and_columns(self):
        import tkinter
        from Deployement import gui
        root = self._root()
        console = self._console(root)
        try:
            self.assertIn("SHADOW", console.banner_mode_var.get())
            cols = tuple(console._tab_approvals.tree.cget("columns"))
            self.assertIn("attempts", cols)
            self.assertEqual(cols[0], "#")  # selection index stable
            self.assertIsNone(console.pipeline)
        finally:
            root.update_idletasks()

    def test_live_switch_requires_confirmation(self):
        from Deployement import gui
        root = self._root()
        asked = []

        class FakeBox:
            @staticmethod
            def askyesno(title, message):
                asked.append((title, message))
                return False  # operator declines

        real_box, gui.messagebox = gui.messagebox, FakeBox
        try:
            console = self._console(root)
            console.mode_var.set("live")
            console._on_mode_change()
            self.assertEqual(console.mode_var.get(), "shadow")
            self.assertFalse(console._live_confirmed)
            self.assertIsNone(console.pipeline)
            self.assertTrue(asked)
        finally:
            gui.messagebox = real_box
            root.update_idletasks()

    def test_display_failure_exits_cleanly(self):
        try:
            import tkinter
            no_display = tkinter.TclError("no display name and no $DISPLAY")
        except Exception:
            self.skipTest("tkinter unavailable")
        from Deployement import app, gui
        real_launch = gui.launch
        try:
            def boom(factory):
                raise no_display
            gui.launch = boom
            code = app.main(["--demo", "--gui", "--mode", "shadow"])
            self.assertEqual(code, 3)
        finally:
            gui.launch = real_launch


class _TorchBlocker:
    """Context manager making `import torch` raise ImportError."""

    def __enter__(self):
        self._saved = sys.modules.get("torch", "ABSENT")
        sys.modules["torch"] = None
        return self

    def __exit__(self, *exc):
        if self._saved == "ABSENT":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved
        return False


class _FakeMarlModule:
    """Injectable stand-in for Marl.mappo.mappo (torch present, Marl not)."""

    def __init__(self, load_behavior="ok"):
        import types
        self.module = types.ModuleType("Marl.mappo.mappo")
        self.load_behavior = load_behavior
        self.instances = []
        outer = self

        class FakeMAPPO:
            def __init__(self, num_host_targets, num_subnet_targets=None):
                self.num_host_targets = num_host_targets
                self.num_subnet_targets = num_subnet_targets
                self.device = "cpu"
                self.eval_called = False
                outer.instances.append(self)

            def load(self, path):
                if outer.load_behavior == "mismatch":
                    raise ValueError("vocab mismatch (fake)")
                import torch
                probe = torch.load(path, map_location="cpu")
                if (probe.get("num_host_targets")
                        != self.num_host_targets):
                    raise ValueError("vocab mismatch (fake)")

            def eval(self):
                self.eval_called = True

            def get_trust_matrix(self):
                raise RuntimeError("no trust in fake")

        self.module.MAPPO = FakeMAPPO
        self._saved = sys.modules.get("Marl.mappo.mappo", "ABSENT")

    def __enter__(self):
        sys.modules["Marl.mappo.mappo"] = self.module
        return self

    def __exit__(self, *exc):
        if self._saved == "ABSENT":
            sys.modules.pop("Marl.mappo.mappo", None)
        else:
            sys.modules["Marl.mappo.mappo"] = self._saved
        return False


class TestModelLoaderErrors(unittest.TestCase):
    def test_missing_torch_is_clean_deployment_error(self):
        from Deployement import model_loader
        with _TorchBlocker():
            with self.assertRaises(model_loader.DeploymentError) as ctx:
                model_loader.load_trained_mappo("whatever.pt")
        # DeploymentError subclasses RuntimeError, NOT ModuleNotFoundError.
        self.assertNotIsInstance(ctx.exception, ModuleNotFoundError)
        message = str(ctx.exception).lower()
        self.assertIn("training", message)
        self.assertIn("venv", message)

    def test_try_imports_reports_reason_not_raise(self):
        from Deployement import model_loader
        with _TorchBlocker():
            ok, reason = model_loader.try_training_imports()
        self.assertFalse(ok)
        self.assertTrue(reason)

    def test_missing_training_venv_is_clean_error(self):
        # torch present but Marl.* unimportable (this machine): the same
        # DeploymentError path as a missing venv, never ModuleNotFoundError.
        import sys
        from Deployement import model_loader
        self.assertNotIn("Marl.mappo.mappo", sys.modules)
        try:
            from Marl.mappo.mappo import MAPPO  # noqa: F401
            self.skipTest("training venv present; cannot simulate absence")
        except Exception:
            pass
        ok, reason = model_loader.try_training_imports()
        self.assertFalse(ok)
        self.assertTrue(reason)  # actionable reason string, not an error
        with self.assertRaises(model_loader.DeploymentError):
            model_loader.load_trained_mappo("whatever.pt")

    def test_garbage_file_rejected(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule(), tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "junk.pt")
            torch.save([1, 2, 3], path)
            with self.assertRaises(model_loader.DeploymentError):
                model_loader.load_trained_mappo(path)

    def test_missing_model_key_rejected(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule(), tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nokeys.pt")
            torch.save({"foo": 1}, path)
            with self.assertRaises(model_loader.DeploymentError):
                model_loader.load_trained_mappo(path)

    def test_old_vocabulary_rejected(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule(), tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.pt")
            torch.save({"model": {}}, path)
            with self.assertRaises(model_loader.DeploymentError) as ctx:
                model_loader.load_trained_mappo(path)
            self.assertIn("vocabulary", str(ctx.exception).lower())

    def test_vocab_mismatch_wrapped_as_deployment_error(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule(load_behavior="mismatch"), \
                tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mm.pt")
            torch.save({"model": {}, "num_host_targets": 137,
                        "num_subnet_targets": 9}, path)
            with self.assertRaises(model_loader.DeploymentError):
                model_loader.load_trained_mappo(path)

    def test_success_path_meta_and_eval(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule() as fake, tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "good.pt")
            torch.save({"model": {}, "num_host_targets": 137,
                        "num_subnet_targets": 9}, path)
            mappo, meta = model_loader.load_trained_mappo(path)
            self.assertEqual(meta["num_host_targets"], 137)
            self.assertEqual(meta["num_subnet_targets"], 9)
            self.assertIsNone(meta["trust"])
            self.assertTrue(mappo.eval_called)

    def test_cli_trained_without_venv_exits_cleanly(self):
        from Deployement import app
        with _TorchBlocker():
            code = app.main(["--policy", "trained",
                             "--checkpoint", "nope.pt",
                             "--headless", "--cycles", "1"])
        self.assertEqual(code, 2)


class TestZoneMatching(unittest.TestCase):
    def test_every_slot_hostname(self):
        from Deployement.asset_map import (match_zone,
                                           zone_slot_hostnames)
        from Deployement.config import AGENT_ZONES
        for zones in AGENT_ZONES.values():
            for zone in zones:
                for cc4 in zone_slot_hostnames(zone):
                    self.assertEqual(match_zone(cc4), zone, cc4)

    def test_routers_explicitly(self):
        from Deployement.asset_map import match_zone
        for zone in ("restricted_zone_a_subnet",
                     "operational_zone_a_subnet",
                     "admin_network_subnet"):
            self.assertEqual(match_zone(f"{zone}_router"), zone)

    def test_internet_host_unmatched(self):
        from Deployement.asset_map import AssetMapError, match_zone
        with self.assertRaises(AssetMapError):
            match_zone("root_internet_host_0")

    def test_empty_and_garbage_raise(self):
        from Deployement.asset_map import AssetMapError, match_zone
        for bad in ("", None, 123, "nosuchhost"):
            with self.assertRaises(AssetMapError, msg=repr(bad)):
                match_zone(bad)

    def test_adversarial_prefix_zones_raise(self):
        from Deployement.asset_map import (AssetMapError,
                                           check_zone_names_unambiguous,
                                           match_zone)
        zones = ["zone_a", "zone_a_b"]
        with self.assertRaises(AssetMapError):
            check_zone_names_unambiguous(zones)
        # And attribution itself never silently picks one:
        with self.assertRaises(AssetMapError):
            match_zone("zone_a_b_host_0", zones)

    def test_unambiguous_custom_zones_ok(self):
        from Deployement.asset_map import (check_zone_names_unambiguous,
                                           match_zone)
        zones = ["zone_a", "zone_b"]
        self.assertTrue(check_zone_names_unambiguous(zones))
        self.assertEqual(match_zone("zone_b_host_3", zones), "zone_b")

    def test_real_subnets_unambiguous(self):
        from Deployement.asset_map import check_zone_names_unambiguous
        from Deployement.config import SUBNETS
        self.assertTrue(check_zone_names_unambiguous(list(SUBNETS)))


class TestAssetValidation(unittest.TestCase):
    def _good_data(self):
        from Deployement.asset_map import generate_default_map
        return generate_default_map().to_dict()

    def test_duplicate_ip_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        hosts0 = data["agents"]["blue_agent_0"]["hosts"]
        hosts1 = data["agents"]["blue_agent_1"]["hosts"]
        hosts1[0]["ip"] = hosts0[0]["ip"]
        with self.assertRaises(AssetMapError) as ctx:
            AssetMap.from_dict(data)
        self.assertIn("duplicate IP", str(ctx.exception))

    def test_invalid_ip_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"][0]["ip"] = "not-an-ip"
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"][0]["ip"] = "999.1.1.1"
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_empty_hostname_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"][0]["hostname"] = "  "
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_invalid_role_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"][0]["role"] = "toaster"
        with self.assertRaises(AssetMapError) as ctx:
            AssetMap.from_dict(data)
        self.assertIn("role", str(ctx.exception))

    def test_router_slot_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        # Appending a router alongside complete slots must hit the
        # router-specific rejection (not just slot completeness).
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"].append(
            {"cc4": "restricted_zone_a_subnet_router",
             "ip": "10.9.9.9", "hostname": "rtr", "role": "router"})
        with self.assertRaises(AssetMapError) as ctx:
            AssetMap.from_dict(data)
        self.assertIn("router", str(ctx.exception).lower())
        # Swapping a real slot for a router is likewise rejected.
        data = self._good_data()
        hosts = data["agents"]["blue_agent_0"]["hosts"]
        hosts[0] = {"cc4": "restricted_zone_a_subnet_router",
                    "ip": "10.9.9.9", "hostname": "rtr",
                    "role": "router"}
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_unknown_agent_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["agents"]["blue_agent_9"] = {"hosts": []}
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_duplicate_cc4_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        dup = dict(data["agents"]["blue_agent_0"]["hosts"][0])
        data["agents"]["blue_agent_1"]["hosts"].append(dup)
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_duplicate_hostname_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"][0]["hostname"] = "same-box"
        data["agents"]["blue_agent_0"]["hosts"][1]["hostname"] = "same-box"
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_unknown_zone_cc4_rejected(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["agents"]["blue_agent_0"]["hosts"][0]["cc4"] = \
            "atlantis_zone_server_host_0"
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)

    def test_unmapped_must_be_objects(self):
        from Deployement.asset_map import AssetMap, AssetMapError
        data = self._good_data()
        data["unmapped"] = ["just-a-string"]
        with self.assertRaises(AssetMapError):
            AssetMap.from_dict(data)


class TestMasksAdversarial(unittest.TestCase):
    # "baa_subnet" CONTAINS "aa_subnet" as a substring, but neither
    # "aa_subnet_" nor "baa_subnet_" prefixes the other: strict
    # attribution separates them where substring matching conflates.
    ZONES = ("aa_subnet", "baa_subnet")

    def test_substring_zones_are_unambiguous(self):
        from Deployement.asset_map import check_zone_names_unambiguous
        self.assertTrue(check_zone_names_unambiguous(list(self.ZONES)))

    def _table(self):
        return [
            {"index": 0, "command": "Monitor", "kind": "none",
             "zone": None, "target": None, "label": "Monitor",
             "asset": None},
            {"index": 1, "command": "BlockTrafficZone", "kind": "zone",
             "zone": "aa_subnet", "target": "aa_subnet", "label": "b-self",
             "asset": None},
            {"index": 2, "command": "BlockTrafficZone", "kind": "zone",
             "zone": "aa_subnet", "target": "baa_subnet",
             "label": "b-other", "asset": None},
        ]

    def _state(self, flagged):
        from Deployement.normalizer import NormalizedHost, NormalizedState
        hosts = {c: NormalizedHost(cc4=c, key="k",
                                   process_event=(c in flagged),
                                   connection_event=False)
                 for c in ("aa_subnet_host_0", "baa_subnet_host_0")}
        return NormalizedState(hosts=hosts, unseen_keys=[])

    def test_substring_zone_isolation(self):
        from Deployement.action_mask import compute_mask
        from Deployement.asset_map import match_zone
        # Sanity: strict attribution separates the pair that substring
        # matching would conflate.
        self.assertEqual(match_zone("baa_subnet_host_0", self.ZONES),
                         "baa_subnet")
        self.assertEqual(match_zone("aa_subnet_host_0", self.ZONES),
                         "aa_subnet")
        # Only aa_subnet visible+quiet: self-block suppressed, but the
        # baa_subnet block (unseen source) stays allowed.
        mask = compute_mask(self._table(), self._state(set()),
                            ("aa_subnet",), match_zones=self.ZONES)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])

    def test_flagged_source_allows(self):
        from Deployement.action_mask import compute_mask
        mask = compute_mask(self._table(),
                            self._state({"aa_subnet_host_0"}),
                            ("aa_subnet",), match_zones=self.ZONES)
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])

    def test_prefix_collision_fails_loudly(self):
        # A zone set where one name prefixes another cannot be
        # attributed strictly: configuration must be rejected, and
        # attribution must raise rather than silently pick.
        from Deployement.asset_map import (AssetMapError,
                                           check_zone_names_unambiguous,
                                           match_zone)
        with self.assertRaises(AssetMapError):
            check_zone_names_unambiguous(["zone_a", "zone_a_b"])
        with self.assertRaises(AssetMapError):
            match_zone("zone_a_b_host_0", ["zone_a", "zone_a_b"])


class _BatchHelper:
    @staticmethod
    def batch(hosts, timestamp=1000.0, notes="", partial_errors=None,
              source=""):
        from Deployement.telemetry import TelemetryBatch
        return TelemetryBatch(timestamp=timestamp, hosts=hosts, notes=notes,
                              source=source,
                              partial_errors=list(partial_errors or []))

    @staticmethod
    def host(processes=None, connections=None, events=None, key="k"):
        from Deployement.telemetry import HostTelemetry
        return HostTelemetry(key=key, processes=list(processes or []),
                             connections=list(connections or []),
                             sessions=[], up=True,
                             events=list(events or []))

    @staticmethod
    def event(kind, severity="low", details=""):
        from Deployement.telemetry import SecurityEvent
        return SecurityEvent(kind=kind, severity=severity, details=details,
                             timestamp=1000.0)


class TestNormalizerHealth(unittest.TestCase):
    def _map(self):
        from Deployement.asset_map import generate_default_map
        return generate_default_map()

    def _norm(self, **kwargs):
        from Deployement.normalizer import TelemetryNormalizer
        return TelemetryNormalizer({"processes": {"sshd"},
                                    "ports": {22},
                                    "peers": {"ctrl"}}, **kwargs)

    def _full_quiet(self, amap):
        hosts = {}
        for agent in amap.agents.values():
            for info in agent["hosts"].values():
                hosts[info["ip"]] = _BatchHelper.host(
                    processes=["sshd"], connections=["tcp:22->ctrl"],
                    key=info["ip"])
        return hosts

    def test_missing_host_is_stale_not_quiet(self):
        amap = self._map()
        norm = self._norm()
        hosts = self._full_quiet(amap)
        victim = next(iter(hosts))
        del hosts[victim]
        state = norm.normalize(_BatchHelper.batch(hosts), amap)
        # Find which cc4 lost telemetry: exactly one stale host family.
        stale = [c for c, h in state.hosts.items()
                 if h.health == "stale"]
        self.assertTrue(stale)
        for cc4 in stale:
            host = state.hosts[cc4]
            self.assertTrue(host.process_event)
            self.assertTrue(host.connection_event)
            self.assertFalse(host.compromised)
        # Everyone else stays quiet.
        for cc4, host in state.hosts.items():
            if cc4 not in stale:
                self.assertEqual(host.health, "quiet", cc4)

    def test_ancient_batch_is_wholly_stale(self):
        amap = self._map()
        norm = self._norm()
        state = norm.normalize(
            _BatchHelper.batch(self._full_quiet(amap), timestamp=1000.0),
            amap, now=100000.0)
        self.assertTrue(all(h.health == "stale"
                            for h in state.hosts.values()))

    def test_partial_error_key_is_stale(self):
        amap = self._map()
        norm = self._norm()
        hosts = self._full_quiet(amap)
        bad_key = next(iter(hosts))
        # Resolve the slot independently of telemetry keys (a failed
        # host's key field stays blank -- only its health speaks).
        bad_cc4 = next(
            c for a in amap.agents.values() for c, info in a["hosts"].items()  # noqa: E501
            if info["ip"] == bad_key)
        state = norm.normalize(
            _BatchHelper.batch(
                hosts, partial_errors=[{"key": bad_key, "error": "boom"}]),
            amap)
        self.assertEqual(state.hosts[bad_cc4].health, "stale")
        self.assertTrue(state.hosts[bad_cc4].process_event)
        self.assertFalse(state.hosts[bad_cc4].compromised)

    def test_compromise_sticky_until_recovery(self):
        amap = self._map()
        norm = self._norm()
        ip = next(iter(self._full_quiet(amap)))
        compromised = _BatchHelper.host(
            processes=["evil"], connections=["tcp:4444->x"],
            events=[_BatchHelper.event("intrusion_confirmed", "critical")],
            key=ip)
        state = norm.normalize(
            _BatchHelper.batch({ip: compromised}), amap)
        cc4 = next(c for c, h in state.hosts.items() if h.key == ip)
        self.assertTrue(state.hosts[cc4].compromised)
        self.assertEqual(state.hosts[cc4].health, "compromised")
        # Later quiet batch: compromise persists (sticky), alerts stay on.
        state2 = norm.normalize(
            _BatchHelper.batch({ip: _BatchHelper.host(key=ip)},
                               timestamp=1001.0), amap)
        self.assertTrue(state2.hosts[cc4].compromised)
        # Explicit recovery clears it.
        rec = _BatchHelper.host(
            events=[_BatchHelper.event("recovery_confirmed")], key=ip)
        state3 = norm.normalize(
            _BatchHelper.batch({ip: rec}, timestamp=1002.0), amap)
        self.assertFalse(state3.hosts[cc4].compromised)
        # ...as does the recovered= set and reset().
        state4 = norm.normalize(
            _BatchHelper.batch({ip: compromised}, timestamp=1003.0), amap)
        self.assertTrue(state4.hosts[cc4].compromised)
        state5 = norm.normalize(
            _BatchHelper.batch({ip: _BatchHelper.host(key=ip)},
                               timestamp=1004.0),
            amap, recovered=[cc4])
        self.assertFalse(state5.hosts[cc4].compromised)
        norm.reset()
        norm2_state = norm.normalize(
            _BatchHelper.batch({ip: _BatchHelper.host(key=ip)},
                               timestamp=1005.0), amap)
        self.assertFalse(norm2_state.hosts[cc4].compromised)

    def test_recovery_after_telemetry_resumes(self):
        amap = self._map()
        norm = self._norm(stale_after_s=10)
        ip = next(iter(self._full_quiet(amap)))
        # Fresh compromise, then telemetry gap (stale, sticky kept)...
        norm.normalize(_BatchHelper.batch(
            {ip: _BatchHelper.host(
                events=[_BatchHelper.event("intrusion_confirmed",
                                           "critical")], key=ip)},
            timestamp=1000.0), amap)
        gap = norm.normalize(_BatchHelper.batch({}, timestamp=1050.0),
                             amap)
        stale_any = [h for h in gap.hosts.values()
                     if h.health == "stale"]
        self.assertTrue(stale_any)
        # Stale keeps last-known compromise (never silently healthy):
        # exactly the previously-compromised host stays compromised.
        still = [c for c, h in gap.hosts.items()
                 if h.health == "stale" and h.compromised]
        self.assertEqual(len(still), 1)
        # ...then telemetry resumes with recovery: clean again.
        rec = norm.normalize(_BatchHelper.batch(
            {ip: _BatchHelper.host(
                processes=["sshd"],
                events=[_BatchHelper.event("recovery_confirmed")], key=ip)},
            timestamp=1060.0), amap)
        cc4r = next(c for c, h in rec.hosts.items() if h.key == ip)
        self.assertFalse(rec.hosts[cc4r].compromised)
        self.assertEqual(rec.hosts[cc4r].health, "quiet")


class TestCompromisePath(unittest.TestCase):
    def test_compromise_without_events_still_matters(self):
        # Item 7 regression: explicit compromise=true with BOTH alert
        # bits false must NOT look clean to mask/policy inputs.
        from Deployement.action_mask import compute_mask
        from Deployement.action_table import build_action_table
        from Deployement.asset_map import generate_default_map
        from Deployement.normalizer import (NormalizedHost, NormalizedState,
                                            TelemetryNormalizer)
        from Deployement.observation import ObservationBuilder
        from Deployement.state_builder import CC4State
        amap = generate_default_map()
        norm = TelemetryNormalizer({"processes": {"sshd"}, "ports": {22},
                                    "peers": {"ctrl"}})
        ip = amap.agents["blue_agent_0"]["hosts"][
            "restricted_zone_a_subnet_server_host_0"]["ip"]
        from Deployement.telemetry import (HostTelemetry, SecurityEvent,
                                           TelemetryBatch)
        batch_hosts = {ip: HostTelemetry(
            key=ip, processes=[], connections=[], sessions=[], up=True,
            events=[SecurityEvent(kind="intrusion_confirmed",
                                  severity="critical")] )}
        state = norm.normalize(TelemetryBatch(timestamp=1.0,
                                              hosts=batch_hosts), amap)
        host = state.hosts["restricted_zone_a_subnet_server_host_0"]
        self.assertTrue(host.compromised)
        # Promotion: alert bits forced on so mask/obs respond.
        self.assertTrue(host.process_event)
        self.assertTrue(host.connection_event)
        table = build_action_table(0, amap)
        mask = compute_mask(table, state, ("restricted_zone_a_subnet",))
        restore = next(i for i, e in enumerate(table)
                       if e["command"] == "Restore" and e["target"] ==
                       "restricted_zone_a_subnet_server_host_0")
        self.assertTrue(mask[restore])
        vec = ObservationBuilder(amap).build_padded(
            0, CC4State(mission_phase=0, blocks={}, hosts=state.hosts))
        proc = vec[1 + 27:1 + 27 + 16]
        self.assertEqual(proc[0], 1)


class TestBaselines(unittest.TestCase):
    def _one_host(self):
        from Deployement.asset_map import generate_default_map
        amap = generate_default_map()
        cc4 = "restricted_zone_a_subnet_server_host_0"
        ip = amap.agents["blue_agent_0"]["hosts"][cc4]["ip"]
        return amap, cc4, ip

    def test_host_override(self):
        from Deployement.normalizer import TelemetryNormalizer
        amap, cc4, ip = self._one_host()
        norm = TelemetryNormalizer(
            {"processes": set(), "hosts": {ip: {"processes": {"weird"}}}})
        state = norm.normalize(_BatchHelper.batch(
            {ip: _BatchHelper.host(processes=["weird"], key=ip)}), amap)
        self.assertFalse(state.hosts[cc4].process_event)

    def test_role_override(self):
        from Deployement.normalizer import TelemetryNormalizer
        amap, cc4, ip = self._one_host()  # role == "server" in default map
        norm = TelemetryNormalizer(
            {"processes": set(),
             "roles": {"server": {"processes": {"srvproc"}}}})
        state = norm.normalize(_BatchHelper.batch(
            {ip: _BatchHelper.host(processes=["srvproc"], key=ip)}), amap)
        self.assertFalse(state.hosts[cc4].process_event)

    def test_precedence_host_over_role_over_global(self):
        from Deployement.normalizer import TelemetryNormalizer
        amap, cc4, ip = self._one_host()
        norm = TelemetryNormalizer(
            {"processes": {"globalproc"},
             "roles": {"server": {"processes": {"roleproc"}}},
             "hosts": {ip: {"processes": {"hostproc"}}}})
        for proc, expect_event in (("hostproc", False), ("roleproc", True),
                                   ("globalproc", True)):
            state = norm.normalize(_BatchHelper.batch(
                {ip: _BatchHelper.host(processes=[proc], key=ip)}), amap)
            self.assertEqual(state.hosts[cc4].process_event, expect_event,
                             proc)

    def test_zone_override(self):
        from Deployement.normalizer import TelemetryNormalizer
        amap, cc4, ip = self._one_host()
        norm = TelemetryNormalizer(
            {"processes": set(),
             "zones": {"restricted_zone_a_subnet":
                       {"processes": {"zoneproc"}}}})
        state = norm.normalize(_BatchHelper.batch(
            {ip: _BatchHelper.host(processes=["zoneproc"], key=ip)}), amap)
        self.assertFalse(state.hosts[cc4].process_event)

    def test_unknown_key_unseen_and_safe(self):
        from Deployement.normalizer import TelemetryNormalizer
        amap, _, _ = self._one_host()
        norm = TelemetryNormalizer({})
        state = norm.normalize(_BatchHelper.batch(
            {"ghost": _BatchHelper.host(processes=["x"], key="ghost")}),
            amap)
        self.assertIn("ghost", state.unseen_keys)
        # Unknown hosts never enter the model-bound host set.
        self.assertNotIn("ghost", state.hosts)


class _LocalHTTPServer:
    """In-process HTTP server for live-collector tests (stdlib only)."""

    def __init__(self, handler):
        import threading
        from http.server import HTTPServer
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self.url = (f"http://127.0.0.1:"
                    f"{self._server.server_address[1]}/hosts")
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.01},
                                        daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()
        return False


def _ok_handler(payload_bytes, code=200, delay=0.0):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if delay:
                import time
                time.sleep(delay)
            body = payload_bytes
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return Handler


def _translate_ok(payload):
    from Deployement.telemetry import HostTelemetry
    out = {}
    for key, item in payload.get("hosts", {}).items():
        out[key] = HostTelemetry(key=key,
                                 processes=list(item.get("processes", [])),
                                 connections=list(item.get("connections",
                                                           [])))
    return out


class TestLiveCollector(unittest.TestCase):
    def _config(self, url, **over):
        from Deployement.live import EndpointConfig
        params = {"name": "test", "endpoint": url, "timeout_s": 5.0}
        params.update(over)
        return EndpointConfig(**params)

    def test_success_path(self):
        import json
        from Deployement.live import HttpJsonCollector
        payload = json.dumps(
            {"hosts": {"h1": {"processes": ["sshd"]}}}).encode()
        with _LocalHTTPServer(_ok_handler(payload)) as server:
            collector = HttpJsonCollector(self._config(server.url),
                                          _translate_ok)
            batch = collector.next_batch()
        self.assertIn("h1", batch.hosts)
        self.assertEqual(batch.source, "test")
        self.assertEqual(batch.partial_errors, [])

    def test_http_error_is_loud(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        with _LocalHTTPServer(_ok_handler(b"nope", code=500)) as server:
            collector = HttpJsonCollector(self._config(server.url),
                                          _translate_ok)
            with self.assertRaises(CollectorError) as ctx:
                collector.next_batch()
        self.assertIn("500", str(ctx.exception))

    def test_malformed_json_is_loud(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        with _LocalHTTPServer(_ok_handler(b"{not json")) as server:
            collector = HttpJsonCollector(self._config(server.url),
                                          _translate_ok)
            with self.assertRaises(CollectorError):
                collector.next_batch()

    def test_timeout_is_loud(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        with _LocalHTTPServer(_ok_handler(b"{}", delay=2.0)) as server:
            collector = HttpJsonCollector(
                self._config(server.url, timeout_s=0.2), _translate_ok)
            with self.assertRaises(CollectorError) as ctx:
                collector.next_batch()
        self.assertIn("ime", str(ctx.exception))  # timeout/timed out

    def test_bad_construction_rejected(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        with self.assertRaises(CollectorError):
            HttpJsonCollector(self._config("", timeout_s=5.0),
                              _translate_ok)
        with self.assertRaises(CollectorError):
            HttpJsonCollector(self._config("http://x", timeout_s=0),
                              _translate_ok)
        with self.assertRaises(CollectorError):
            HttpJsonCollector(self._config("http://x"), "not-callable")

    def test_translator_contract_enforced(self):
        import json
        from Deployement.live import CollectorError, HttpJsonCollector
        payload = json.dumps({"hosts": {}}).encode()
        with _LocalHTTPServer(_ok_handler(payload)) as server:
            with self.assertRaises(CollectorError):
                HttpJsonCollector(self._config(server.url),
                                  lambda p: ["not", "a", "dict"]
                                  ).next_batch()

    def test_partial_host_failure(self):
        import json
        from Deployement.live import (HostTranslateError, HttpJsonCollector)

        def flaky(payload):
            return {"good": _translate_ok(payload)["good"]
                    if "good" in payload.get("hosts", {}) else None,
                    "bad": HostTranslateError("unparseable row")}

        payload = json.dumps({"hosts": {"good": {"processes": []}}}).encode()
        with _LocalHTTPServer(_ok_handler(payload)) as server:
            batch = HttpJsonCollector(self._config(server.url),
                                      flaky).next_batch()
        self.assertIn("good", batch.hosts)
        self.assertEqual(len(batch.partial_errors), 1)
        self.assertEqual(batch.partial_errors[0]["key"], "bad")

    def test_missing_token_names_variable(self):
        from Deployement.live import (EndpointConfig, HttpJsonCollector,
                                      SecretMissingError, resolve_secret)
        with self.assertRaises(SecretMissingError) as ctx:
            resolve_secret("DEPLOY_DEFINITELY_UNSET_TOKEN_XYZ", "test")
        self.assertIn("DEPLOY_DEFINITELY_UNSET_TOKEN_XYZ",
                      str(ctx.exception))
        # ...and a collector naming a missing var fails at construction.
        cfg = EndpointConfig(name="t", endpoint="http://x",
                             token_env="DEPLOY_DEFINITELY_UNSET_TOKEN_XYZ")
        with self.assertRaises(SecretMissingError):
            HttpJsonCollector(cfg, _translate_ok)

    def test_token_from_env_not_config(self):
        import json
        import os
        from Deployement.live import HttpJsonCollector
        os.environ["DEPLOY_TEST_TOKEN_XYZ"] = "s3cret-value"
        self.addCleanup(os.environ.pop, "DEPLOY_TEST_TOKEN_XYZ", None)
        payload = json.dumps({"hosts": {}}).encode()
        seen = {}

        from http.server import BaseHTTPRequestHandler

        class AuthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["auth"] = self.headers.get("Authorization")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        with _LocalHTTPServer(AuthHandler) as server:
            cfg = self._config(server.url,
                               token_env="DEPLOY_TEST_TOKEN_XYZ")
            batch = HttpJsonCollector(cfg, _translate_ok).next_batch()
        self.assertEqual(batch.hosts, {})
        self.assertEqual(seen.get("auth"), "Bearer s3cret-value")

    def test_resilient_retries_then_succeeds(self):
        from Deployement.live import CollectorError, ResilientCollector
        from Deployement.telemetry import TelemetryBatch
        calls = []

        class Flaky:
            exhausted = False

            def next_batch(self):
                calls.append(1)
                if len(calls) < 3:
                    raise CollectorError("transient")
                return TelemetryBatch(timestamp=1.0, hosts={})

            def close(self):
                pass

        batch = ResilientCollector(Flaky(), retries=3,
                                   backoff_s=0).next_batch()
        self.assertEqual(batch.hosts, {})
        self.assertEqual(len(calls), 3)

    def test_poll_interval_respected(self):
        # Fake clock/sleeper: consecutive polls honor poll_interval_s
        # with zero real waiting.
        import json
        from Deployement.live import HttpJsonCollector
        now = [1000.0]
        slept = []

        def fake_clock():
            return now[0]

        def fake_sleeper(seconds):
            slept.append(seconds)
            now[0] += seconds

        payload = json.dumps({"hosts": {}}).encode()
        with _LocalHTTPServer(_ok_handler(payload)) as server:
            collector = HttpJsonCollector(
                self._config(server.url, poll_interval_s=5.0),
                _translate_ok, clock=fake_clock, sleeper=fake_sleeper)
            collector.next_batch()          # first poll: immediate
            self.assertEqual(slept, [])
            now[0] += 2.0                   # only 2s elapsed...
            collector.next_batch()          # ...so ~3s remainder slept
            self.assertEqual(len(slept), 1)
            self.assertAlmostEqual(slept[0], 3.0, places=6)
            now[0] += 10.0                  # interval already covered...
            collector.next_batch()          # ...so no sleep at all
            self.assertEqual(len(slept), 1)

    def test_poll_interval_zero_never_sleeps(self):
        import json
        from Deployement.live import HttpJsonCollector
        slept = []
        payload = json.dumps({"hosts": {}}).encode()
        with _LocalHTTPServer(_ok_handler(payload)) as server:
            collector = HttpJsonCollector(
                self._config(server.url, poll_interval_s=0),
                _translate_ok,
                clock=lambda: 0.0, sleeper=slept.append)
            collector.next_batch()
            collector.next_batch()
        self.assertEqual(slept, [])

    def test_negative_poll_interval_rejected(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        with self.assertRaises(CollectorError):
            HttpJsonCollector(
                self._config("http://127.0.0.1:9", poll_interval_s=-1),
                _translate_ok)

    def test_oversized_response_rejected_before_parsing(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        big = b'{"hosts": {"h": "' + b"x" * 5000 + b'"}}'
        with _LocalHTTPServer(_ok_handler(big)) as server:
            collector = HttpJsonCollector(
                self._config(server.url, max_response_bytes=100),
                _translate_ok)
            with self.assertRaises(CollectorError) as ctx:
                collector.next_batch()
        self.assertIn("max_response_bytes", str(ctx.exception))

    def test_response_at_cap_is_accepted(self):
        import json
        from Deployement.live import HttpJsonCollector
        payload = json.dumps({"hosts": {"h1": {}}}).encode()
        with _LocalHTTPServer(_ok_handler(payload)) as server:
            collector = HttpJsonCollector(
                self._config(server.url,
                             max_response_bytes=len(payload)),
                _translate_ok)
            batch = collector.next_batch()
        self.assertIn("h1", batch.hosts)

    def test_bad_max_response_bytes_rejected(self):
        from Deployement.live import CollectorError, HttpJsonCollector
        with self.assertRaises(CollectorError):
            HttpJsonCollector(
                self._config("http://127.0.0.1:9", max_response_bytes=0),
                _translate_ok)
        with self.assertRaises(CollectorError):
            HttpJsonCollector(
                self._config("http://127.0.0.1:9", max_response_bytes=-5),
                _translate_ok)

    def test_resilient_gives_up_loudly(self):
        from Deployement.live import CollectorError, ResilientCollector

        class AlwaysDown:
            exhausted = False

            def next_batch(self):
                raise CollectorError("down")

            def close(self):
                pass

        with self.assertRaises(CollectorError):
            ResilientCollector(AlwaysDown(), retries=1,
                               backoff_s=0).next_batch()

    def test_scrub_secrets(self):
        from Deployement.live import scrub_secrets
        dirty = {"api_token": "abc", "nested": {"password": "x"},
                 "hosts": ["h1"], "count": 3,
                 "note": "nothing secret here"}
        clean = scrub_secrets(dirty)
        self.assertEqual(clean["api_token"], "***REDACTED***")
        self.assertEqual(clean["nested"]["password"], "***REDACTED***")
        self.assertEqual(clean["hosts"], ["h1"])
        self.assertEqual(clean["count"], 3)


class _StubMappo:
    """Minimal MAPPO-shaped stub (no Marl/training deps)."""

    def __init__(self, num_host_targets=137, mode="ok"):
        self.num_host_targets = num_host_targets
        self.device = "cpu"
        self.eval_called = False
        self.mode = mode

    def eval(self):
        self.eval_called = True

    def _prepare_host_active_mask(self, mask, batch):
        import numpy as np
        return mask

    def get_messages_for_agent(self, receiver_id, messages):
        import torch
        return torch.zeros(5, 128)

    def get_trust_for_agent(self, receiver_id):
        import torch
        return torch.full((5,), 0.5)

    def actor_forward(self, observations, action_masks=None,
                      received_messages=None, trust_weights=None,
                      host_active_mask=None):
        import torch
        assert not torch.is_grad_enabled(), \
            "inference built an autograd graph"
        logits = torch.zeros(242)
        if self.mode == "nonfinite":
            logits[:] = float("inf")
        elif self.mode == "oor":
            logits = torch.zeros(300)
            logits[299] = 10.0
        else:
            logits[16] = 10.0  # Monitor for agent 0's table
        return logits

    def get_outgoing_messages(self, obs, return_decoded=True,
                              host_active_mask=None, host_valid_mask=None):
        import torch
        messages = [{"event_type": 0, "target_type": 0, "target_id": 0,
                     "threat_level": 0, "status": 0, "priority": 0,
                     "confidence": 0.0} for _ in range(5)]
        return torch.zeros(5, 128), messages


class TestTrainedInference(unittest.TestCase):
    def _tables_obs_masks(self):
        import numpy as np
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        amap = generate_default_map()
        tables = build_all_tables(amap)
        obs = np.zeros((5, 210), dtype=np.float32)
        masks = [np.ones(len(tables[i]), dtype=bool) for i in range(5)]
        return tables, obs, masks

    def test_eval_called_and_contract_ok(self):
        from Deployement.policy_engine import TrainedPolicyEngine
        tables, _, _ = self._tables_obs_masks()
        stub = _StubMappo(num_host_targets=137)
        engine = TrainedPolicyEngine(stub, tables, 137)
        self.assertTrue(stub.eval_called)

    def test_vocab_mismatch_rejected(self):
        from Deployement.policy_engine import TrainedPolicyEngine
        tables, _, _ = self._tables_obs_masks()
        with self.assertRaises(ValueError):
            TrainedPolicyEngine(_StubMappo(num_host_targets=95),
                                tables, 137)
        with self.assertRaises(ValueError):
            TrainedPolicyEngine(_StubMappo(num_host_targets=137),
                                tables, 96)

    def test_decide_no_grad_and_valid(self):
        import numpy as np
        from Deployement.policy_engine import TrainedPolicyEngine
        tables, obs, masks = self._tables_obs_masks()
        engine = TrainedPolicyEngine(_StubMappo(), tables, 137)
        host_masks = np.zeros((5, 3, 16), dtype=bool)
        host_valid = np.ones((5, 137), dtype=bool)
        decisions = engine.decide(obs, masks, host_masks=host_masks,
                                  host_valid=host_valid)
        self.assertEqual(len(decisions), 5)
        for record in decisions:
            self.assertIn(record["action"], range(242))
            self.assertEqual(len(record["trust_row"]), 5)

    def test_nonfinite_logits_refused(self):
        import numpy as np
        from Deployement.policy_engine import TrainedPolicyEngine
        tables, obs, masks = self._tables_obs_masks()
        engine = TrainedPolicyEngine(_StubMappo(mode="nonfinite"),
                                     tables, 137)
        with self.assertRaises(RuntimeError):
            engine.decide(obs, masks)

    def test_out_of_range_action_refused(self):
        import numpy as np
        from Deployement.policy_engine import TrainedPolicyEngine
        tables, obs, masks = self._tables_obs_masks()
        engine = TrainedPolicyEngine(_StubMappo(mode="oor"), tables, 137)
        with self.assertRaises(RuntimeError):
            engine.decide(obs, masks)


class TestValidatorExtras(unittest.TestCase):
    def _setup(self, **kwargs):
        from Deployement.action_table import build_action_table
        from Deployement.asset_map import generate_default_map
        from Deployement.validator import ActionValidator
        amap = generate_default_map()
        table = build_action_table(0, amap)
        mask = np.ones(len(table), dtype=bool)
        params = {"mode": "mock", "cooldown_s": 60}
        params.update(kwargs)
        return ActionValidator(**params), table, mask

    def test_reset_clears_cooldowns(self):
        validator, table, mask = self._setup()
        restore = next(i for i, e in enumerate(table)
                       if e["command"] == "Restore")
        validator.validate(0, restore, table, mask, now=1000.0)
        with self.assertRaises(Exception):
            validator.validate(0, restore, table, mask, now=1050.0)
        validator.reset()
        decision = validator.validate(0, restore, table, mask, now=1050.0)
        self.assertTrue(decision["simulated"])

    def test_approve_stamps_approval_time(self):
        validator, table, mask = self._setup(mode="supervised")
        restore = next(i for i, e in enumerate(table)
                       if e["command"] == "Restore")
        queued = validator.validate(0, restore, table, mask, now=1000.0)
        approved = validator.approve(queued, now=2000.0)
        self.assertEqual(approved["approved_at"], 2000.0)
        # Cooldown runs from approval (2000), not decision (1000):
        # 2050 is inside the 300s window...
        with self.assertRaises(Exception):
            validator.validate(0, restore, table, mask, now=2050.0)
        # ...and 2400 is outside it.
        validator.validate(0, restore, table, mask, now=2400.0)

    def test_live_marks_enforce(self):
        validator, table, mask = self._setup(mode="live")
        monitor = next(i for i, e in enumerate(table)
                       if e["command"] == "Monitor")
        auto = validator.validate(0, monitor, table, mask)
        self.assertTrue(auto["enforce"])
        self.assertFalse(auto["simulated"])
        restore = next(i for i, e in enumerate(table)
                       if e["command"] == "Restore")
        queued = validator.validate(0, restore, table, mask)
        self.assertTrue(queued["needs_approval"])
        self.assertFalse(queued["enforce"])
        approved = validator.approve(queued, now=5000.0)
        self.assertTrue(approved["enforce"])

    def test_idempotency_key_present_and_stable(self):
        validator, table, mask = self._setup()
        monitor = next(i for i, e in enumerate(table)
                       if e["command"] == "Monitor")
        first = validator.validate(0, monitor, table, mask, now=100.0)
        second = validator.validate(0, monitor, table, mask, now=100.0)
        self.assertIn("idempotency_key", first)
        self.assertEqual(first["idempotency_key"],
                         second["idempotency_key"])

    def test_idempotency_key_scoped_by_context(self):
        # Same wall-clock second, different sessions/cycles must NOT
        # share a key, or backends would suppress distinct intents.
        validator, table, mask = self._setup()
        monitor = next(i for i, e in enumerate(table)
                       if e["command"] == "Monitor")
        first = validator.validate(0, monitor, table, mask, now=100.0,
                                   key_context="sessA:3")
        second = validator.validate(0, monitor, table, mask, now=100.0,
                                    key_context="sessA:4")
        third = validator.validate(0, monitor, table, mask, now=100.0,
                                   key_context="sessA:3")
        self.assertNotEqual(first["idempotency_key"],
                            second["idempotency_key"])
        self.assertEqual(first["idempotency_key"],
                         third["idempotency_key"])
        # Legacy callers without context keep the unscoped format.
        legacy = validator.validate(0, monitor, table, mask, now=100.0)
        self.assertFalse(legacy["idempotency_key"].startswith(":"))


class TestExecutors(unittest.TestCase):
    def test_null_backend_refuses_everything(self):
        from Deployement.executor import BackendUnavailable, NullBackend
        backend = NullBackend()
        calls = [
            lambda: backend.collect_status({"ip": "1.1.1.1"}, 5, "k1"),
            lambda: backend.collect_forensics({"ip": "1.1.1.1"}, 5, "k2"),
            lambda: backend.deploy_honeypot("h", 5, "k3"),
            lambda: backend.terminate_suspicious("h", 5, "k4"),
            lambda: backend.reimage_host("h", 5, "k5"),
            lambda: backend.set_zone_block("a", "b", 5, "k6"),
            lambda: backend.clear_zone_block("a", "b", 5, "k7"),
        ]
        for call in calls:
            with self.assertRaises(BackendUnavailable):
                call()

    def test_mock_backend_idempotent(self):
        from Deployement.executor import EnforcementState, MockBackend
        state = EnforcementState()
        backend = MockBackend(state)
        first = backend.set_zone_block("src", "dst", 30, "key-1")
        second = backend.set_zone_block("src", "dst", 30, "key-1")
        self.assertTrue(first.applied)
        self.assertTrue(second.applied)  # deduped single effect
        self.assertEqual(state.blocks, {"dst": {"src"}})
        self.assertEqual(len(state.blocks["dst"]), 1)
        self.assertEqual(len(backend.audit), 2)
        self.assertTrue(backend.audit[1]["duplicate"])

    def test_failed_attempts_do_not_poison_idempotency(self):
        # A failed attempt must not be cached: the retry has to really
        # execute instead of reporting "duplicate suppressed" success.
        from Deployement.executor import (BackendResult, EnforcementState,
                                          MockBackend)
        state = EnforcementState()
        backend = MockBackend(state)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                return BackendResult(applied=False, details="nope",
                                     error="boom")
            return BackendResult(applied=True, details="ok")

        first, _ = backend._once("k-flaky", flaky)
        self.assertFalse(first.applied)
        second, _ = backend._once("k-flaky", flaky)
        self.assertTrue(second.applied)
        self.assertEqual(len(calls), 2)
        # ...while genuine duplicates of a SUCCESS are still suppressed.
        third, _ = backend._once("k-flaky", flaky)
        self.assertTrue(third.applied)
        self.assertEqual(len(calls), 2)

    def test_mock_backend_rollback_roundtrip(self):
        from Deployement.executor import (EnforcementState, MockBackend,
                                          RollbackUnsupported)
        state = EnforcementState()
        backend = MockBackend(state)
        backend.set_zone_block("src", "dst", 30, "k")
        self.assertIn("dst", state.blocks)
        record = next(r for r in backend.audit if r.get("undo"))
        backend.rollback(record)
        self.assertNotIn("dst", state.blocks)
        backend.reimage_host("h", 30, "k2")
        no_undo = next(r for r in backend.audit
                       if r.get("details", "").startswith("mock reimage"))
        with self.assertRaises(RollbackUnsupported):
            backend.rollback(no_undo)

    def test_live_executor_queues_then_applies(self):
        from Deployement.executor import (EnforcementState, LiveExecutor,
                                          MockBackend)
        state = EnforcementState()
        executor = LiveExecutor(state, backend=MockBackend(state))
        decision = {"operation": "isolate_zone_traffic", "zone": "dst",
                    "target": "src", "needs_approval": True,
                    "approved": False, "agent_id": 0, "command": "Block",
                    "idempotency_key": "q1"}
        queued = executor.execute(decision)
        self.assertFalse(queued.applied)
        self.assertIn("approval", queued.details)
        self.assertEqual(len(state.pending_approvals), 1)
        decision["approved"] = True
        applied = executor.execute(decision)
        self.assertTrue(applied.applied)
        self.assertIn("dst", state.blocks)

    def test_live_executor_failure_never_silent(self):
        from Deployement.executor import EnforcementState, LiveExecutor
        state = EnforcementState()
        executor = LiveExecutor(state)  # NullBackend by default
        decision = {"operation": "reimage_host", "zone": "z",
                    "target": "h", "needs_approval": True,
                    "approved": True, "agent_id": 0,
                    "command": "Restore", "idempotency_key": "q9"}
        result = executor.execute(decision)
        self.assertFalse(result.applied)
        self.assertTrue(result.error)
        failed = [a for a in state.audit if a.get("event") == "failed"]
        self.assertTrue(failed)

    def test_shadow_and_mock_audit(self):
        from Deployement.executor import (EnforcementState, MockExecutor,
                                          ShadowExecutor)
        state = EnforcementState()
        ShadowExecutor(state).execute({"operation": "reimage_host",
                                       "target": "h"})
        MockExecutor(state).execute({"operation": "reimage_host",
                                     "zone": "z", "target": "h"})
        kinds = [a["event"] for a in state.audit]
        self.assertIn("shadowed", kinds)
        self.assertIn("executed", kinds)
        self.assertIn("h", state.remediated_hosts)


class _ScriptedCollector:
    """Yields scripted batches/errors in order (pipeline failure tests)."""

    def __init__(self, script):
        self._script = list(script)
        self._pos = 0

    @property
    def exhausted(self):
        return self._pos >= len(self._script)

    def next_batch(self):
        if self._pos >= len(self._script):
            return None
        item = self._script[self._pos]
        self._pos += 1
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


class _FixedPolicy:
    """Policy stub returning fixed per-agent action indices."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.reset_called = 0

    def decide(self, obs_batch, masks, host_masks=None, host_valid=None):
        return [{"action": int(a), "probs_top": [],
                 "message": None, "trust_row": []}
                for a in self.actions]

    def reset(self):
        self.reset_called += 1


class _RaisingPolicy(_FixedPolicy):
    def decide(self, *args, **kwargs):
        raise RuntimeError("policy exploded")


class _RaisingExecutor:
    name = "raising"

    def __init__(self, state=None):
        self.state = state

    def execute(self, decision):
        raise RuntimeError("executor exploded")


class TestPipelineSafety(unittest.TestCase):
    def _pipeline(self, **over):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        params = {"mode": "shadow", "baselines": demo_baselines()}
        params.update(over)
        return DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)), **params), amap

    def test_reset_isolation(self):
        from Deployement.live import CollectorHealth
        pipeline, amap = self._pipeline(mode="mock", cooldown_s=60)
        # Force a destructive, cooldown-tracked decision through.
        tables = pipeline.tables
        block0 = next(i for i, e in enumerate(tables[0])
                      if e["command"] == "BlockTrafficZone")
        pipeline.policy = _FixedPolicy([block0, 16, 16, 16, 48])
        pipeline.run(max_cycles=1)
        self.assertTrue(pipeline.enforcement.blocks)
        first_session = pipeline.session_id
        first_cycle = pipeline.cycle
        self.assertGreater(first_cycle, 0)
        # Cooldown state exists now: immediate re-fire would be refused.
        import numpy as _np
        full_mask = _np.ones(len(tables[0]), dtype=bool)
        with self.assertRaises(Exception):
            pipeline.validator.validate(0, block0, tables[0], full_mask)
        pipeline.reset()
        self.assertEqual(pipeline.cycle, 0)
        self.assertEqual(pipeline.history, [])
        self.assertEqual(pipeline.enforcement.blocks, {})
        self.assertEqual(pipeline.enforcement.pending_approvals, [])
        self.assertNotEqual(pipeline.session_id, first_session)
        self.assertIsInstance(pipeline.health, CollectorHealth)
        self.assertEqual(pipeline.health.consecutive_failures, 0)
        # Cooldowns cleared: same destructive fires again immediately.
        pipeline.validator.validate(0, block0, tables[0], full_mask)
        pipeline.close()

    def test_normalizer_memory_cleared_on_reset(self):
        pipeline, amap = self._pipeline()
        ip = amap.agents["blue_agent_0"]["hosts"][
            "restricted_zone_a_subnet_server_host_0"]["ip"]
        from Deployement.telemetry import (HostTelemetry, SecurityEvent,
                                           TelemetryBatch)
        evil = {ip: HostTelemetry(
            key=ip, events=[SecurityEvent(
                kind="intrusion_confirmed", severity="critical")])}
        pipeline.normalizer.normalize(TelemetryBatch(timestamp=1.0,
                                                     hosts=evil), amap)
        cc4 = "restricted_zone_a_subnet_server_host_0"
        self.assertTrue(
            pipeline.normalizer._last_compromised.get(cc4, False))
        pipeline.reset()
        self.assertEqual(pipeline.normalizer._last_compromised, {})
        self.assertEqual(pipeline.normalizer._last_seen, {})

    def test_collector_failure_synthesizes_stale(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.live import CollectorError
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        script = [CollectorError("link down"), CollectorError("still down")]
        pipeline = DeploymentPipeline(
            amap, _ScriptedCollector(script),
            MockPolicyEngine(build_all_tables(amap)), mode="shadow",
            baselines=None)
        records = pipeline.run()
        pipeline.close()
        self.assertEqual(len(records), 10)  # 2 cycles x 5 agents
        self.assertEqual(pipeline.health.state, "degraded")
        for record in records:
            self.assertEqual(len(record["stale_hosts"]), 112)
            self.assertEqual(record["health"], "degraded")
        # Third consecutive failure trips "down".
        pipeline2 = DeploymentPipeline(
            amap, _ScriptedCollector([CollectorError("x")] * 3),
            MockPolicyEngine(build_all_tables(amap)), mode="shadow",
            baselines=None)
        pipeline2.run()
        pipeline2.close()
        self.assertEqual(pipeline2.health.state, "down")

    def test_none_from_live_collector_is_failure(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()

        class NoneLive:
            exhausted = False

            def next_batch(self):
                return None

            def close(self):
                pass

        pipeline = DeploymentPipeline(
            amap, NoneLive(), MockPolicyEngine(build_all_tables(amap)),
            mode="shadow", baselines=None, stale_after_s=300)
        records = pipeline.step()
        self.assertEqual(len(records), 5)
        self.assertNotEqual(pipeline.health.state, "ok")
        pipeline.close()

    def test_policy_failure_records_rejected(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_collector
        from Deployement.pipeline import DeploymentPipeline
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap), _RaisingPolicy([]), mode="shadow")
        records = pipeline.step()
        pipeline.close()
        self.assertEqual(len(records), 5)
        for record in records:
            self.assertTrue(record["status"].startswith("rejected"))
            self.assertIn("policy failure", record["status"])
            self.assertFalse(record["validation"]["ok"])
        self.assertEqual(pipeline.cycle, 1)

    def test_executor_failure_records_rejected(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)), mode="shadow",
            baselines=demo_baselines())
        pipeline.executor = _RaisingExecutor(pipeline.enforcement)
        records = pipeline.step()
        pipeline.close()
        self.assertTrue(all(r["status"].startswith("rejected")
                            for r in records))

    def test_supervised_approval_flow_end_to_end(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        amap = generate_default_map()
        tables = build_all_tables(amap)
        restore0 = next(i for i, e in enumerate(tables[0])
                        if e["command"] == "Restore")
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            _FixedPolicy([restore0, 16, 16, 16, 48]), mode="supervised",
            baselines=demo_baselines(), cooldown_s=0)
        # Cycles 0-2: quiet, scan, compromise. Restore is masked while
        # quiet but allowed once alerts appear -- and then QUEUED, never
        # silently applied.
        pipeline.step()  # cycle 0 (quiet): masked -> rejected
        pipeline.step()  # cycle 1 (scan alerts): queued
        self.assertTrue(pipeline.enforcement.pending_approvals)
        for record in pipeline.history:
            self.assertNotEqual(record["status"], "applied")
        result = pipeline.approve_pending(0)
        self.assertTrue(result.applied)
        self.assertIn(
            "restricted_zone_a_subnet_server_host_0",
            pipeline.enforcement.remediated_hosts)
        pipeline.close()

    def test_records_carry_audit_fields(self):
        pipeline, _amap = self._pipeline()
        records = pipeline.step()
        pipeline.close()
        record = records[0]
        for key in ("session_id", "timestamp", "health", "stale_hosts",
                    "unseen_keys", "validation", "approval", "exec"):
            self.assertIn(key, record)
        self.assertEqual(record["session_id"], pipeline.session_id)
        self.assertTrue(record["validation"]["ok"])


class TestAppAndSiteConfig(unittest.TestCase):
    def test_live_requires_explicit_flag(self):
        from Deployement import app
        with self.assertRaises(SystemExit) as ctx:
            app.main(["--mode", "live", "--headless"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_live_mock_backend_headless_runs(self):
        from Deployement import app
        code = app.main(["--mode", "live", "--enable-live",
                         "--backend", "mock", "--demo", "--headless",
                         "--cycles", "1"])
        self.assertEqual(code, 0)

    def test_live_null_backend_fails_loudly_not_silently(self):
        # Default backend in live mode refuses loudly: decisions exist
        # but nothing reports success.
        from Deployement import app
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = app.main(["--mode", "live", "--enable-live",
                             "--demo", "--headless", "--cycles", "1"])
        self.assertEqual(code, 0)  # ran; check nothing fake-succeeded
        out = buf.getvalue()
        self.assertNotIn("(applied)", out)

    def test_invalid_backend_spec_fails_cleanly(self):
        from Deployement import app
        with self.assertRaises(SystemExit) as ctx:
            app.main(["--mode", "live", "--enable-live",
                      "--backend", "not-a-real-backend",
                      "--demo", "--headless", "--cycles", "1"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_live_host_op_receives_real_asset(self):
        # Issue #1 end-to-end: simulated CC4 target -> AssetMap ->
        # real asset reaches the backend; pipeline state stays CC4.
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        amap = generate_default_map()
        tables = build_all_tables(amap)
        restore0 = next(i for i, e in enumerate(tables[0])
                        if e["command"] == "Restore")
        cc4 = "restricted_zone_a_subnet_server_host_0"
        real_ip = amap.agents["blue_agent_0"]["hosts"][cc4]["ip"]
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            _FixedPolicy([restore0, 16, 16, 16, 48]),
            mode="live", baselines=demo_baselines(), cooldown_s=0,
            live_backend="mock")
        # Cycles 0-2 reach the compromise; Restore is masked until
        # alerts appear, then approved+executed via the mock backend.
        pipeline.step()
        pipeline.step()
        self.assertTrue(pipeline.enforcement.pending_approvals)
        result = pipeline.approve_pending(0)
        self.assertTrue(result.applied)
        backend = pipeline.executor.backend
        applied_to = [r for r in backend.audit
                      if r.get("details", "").startswith("mock reimage")]
        self.assertTrue(applied_to)
        # Backend saw the REAL asset...
        self.assertIn(real_ip, applied_to[0]["details"])
        self.assertNotIn(cc4, applied_to[0]["details"])
        # ...while pipeline-facing state stayed in CC4 names.
        self.assertIn(cc4, pipeline.enforcement.remediated_hosts)
        pipeline.close()

    def test_real_asset_resolution_unit(self):
        from Deployement.executor import LiveExecutor
        decide = LiveExecutor._real_asset
        self.assertEqual(
            decide({"asset": {"ip": "10.1.2.3", "hostname": "h"},
                    "target": "cc4_host"}), "10.1.2.3")
        self.assertEqual(
            decide({"asset": {"ip": "", "hostname": "box-1"},
                    "target": "cc4_host"}), "box-1")
        self.assertEqual(decide({"asset": None, "target": "cc4_host"}),
                         "cc4_host")
        self.assertEqual(decide({"asset": {}, "target": "cc4_host"}),
                         "cc4_host")

    def test_executor_timeout_mapping(self):
        from Deployement.config import DEFAULT_OP_TIMEOUT_S
        from Deployement.executor import LiveExecutor, EnforcementState
        executor = LiveExecutor(EnforcementState())
        self.assertGreater(executor._timeout("reimage_host"), 0)
        self.assertEqual(executor._timeout("no_such_op"),
                         float(DEFAULT_OP_TIMEOUT_S))

    def test_site_config_merge_and_env_override(self):
        import json
        import os
        import tempfile
        from Deployement import site_config
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "site.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mode": "mock", "mission_phase": 1,
                           "collector": {"type": "demo"}}, fh)
            cfg = site_config.load_site_config(path)
            self.assertEqual(cfg["mode"], "mock")
            self.assertEqual(cfg["mission_phase"], 1)
            os.environ["DEPLOY_MODE"] = "shadow"
            self.addCleanup(os.environ.pop, "DEPLOY_MODE", None)
            cfg2 = site_config.load_site_config(path)
            self.assertEqual(cfg2["mode"], "shadow")

    def test_site_config_rejects_embedded_secrets(self):
        import json
        import os
        import tempfile
        from Deployement import site_config
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "evil.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"collector": {"token": "hunter2"}}, fh)
            with self.assertRaises(ValueError):
                site_config.load_site_config(path)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"collector": {"endpoint":
                                         "https://user:pass@host/x"}}, fh)
            with self.assertRaises(ValueError):
                site_config.load_site_config(path)

    def test_site_config_rejects_bad_mission_phase(self):
        import json
        import os
        import tempfile
        from Deployement import site_config
        for bad in (3, -1, "late", 1.5):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "site.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"mission_phase": bad}, fh)
                with self.assertRaises(ValueError, msg=repr(bad)):
                    site_config.load_site_config(path)
        # Boundary values load cleanly.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "site.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mission_phase": 2}, fh)
            cfg = site_config.load_site_config(path)
            self.assertEqual(cfg["mission_phase"], 2)

    def test_site_config_token_env_reference_ok(self):
        import json
        import os
        import tempfile
        from Deployement import site_config
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "good.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"collector": {"token_env": "DEPLOY_XYZ"}}, fh)
            cfg = site_config.load_site_config(path)
            self.assertEqual(cfg["collector"]["token_env"], "DEPLOY_XYZ")

    def test_bad_site_config_rejected(self):
        import os
        import tempfile
        from Deployement import site_config
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(ValueError):
                site_config.load_site_config(path)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"mode": "hyperdrive"}')
            with self.assertRaises(ValueError):
                site_config.load_site_config(path)

    def test_file_collector_malformed_line(self):
        import os
        import tempfile
        from Deployement.live import CollectorError
        from Deployement.telemetry import FileCollector
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{bad json\n")
            collector = FileCollector(path)
            with self.assertRaises(CollectorError):
                collector.next_batch()
            collector.close()

    def test_file_collector_exhaustion_flag(self):
        import json
        import os
        import tempfile
        from Deployement.telemetry import FileCollector
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp": 1.0, "hosts": {}}) + "\n")
            collector = FileCollector(path)
            self.assertFalse(collector.exhausted)
            batch = collector.next_batch()
            self.assertEqual(batch.timestamp, 1.0)
            self.assertIsNone(collector.next_batch())
            self.assertTrue(collector.exhausted)
            collector.close()

    def test_file_collector_skips_blank_lines(self):
        import json
        import os
        import tempfile
        from Deployement.telemetry import FileCollector
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n")
                fh.write(json.dumps({"timestamp": 2.0, "hosts": {}}) + "\n")
                fh.write("   \n")
                fh.write(json.dumps({"timestamp": 3.0, "hosts": {}}) + "\n")
                fh.write("\n")  # trailing newline must not raise
            collector = FileCollector(path)
            try:
                first = collector.next_batch()
                self.assertEqual(first.timestamp, 2.0)
                second = collector.next_batch()
                self.assertEqual(second.timestamp, 3.0)
                self.assertIsNone(collector.next_batch())
                self.assertTrue(collector.exhausted)
            finally:
                collector.close()

    def test_missing_assets_file_is_clean_cli_error(self):
        import os
        import tempfile
        from Deployement import app
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.json")
            code = app.main(["--assets", missing, "--demo", "--headless",
                             "--cycles", "1"])
            self.assertEqual(code, 2)

    def test_missing_capture_file_is_clean_cli_error(self):
        import os
        import tempfile
        from Deployement import app
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.jsonl")
            code = app.main(["--collector", missing, "--headless",
                             "--cycles", "1"])
            self.assertEqual(code, 2)

    def test_malformed_assets_file_is_clean_cli_error(self):
        import os
        import tempfile
        from Deployement import app
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            code = app.main(["--assets", path, "--demo", "--headless",
                             "--cycles", "1"])
            self.assertEqual(code, 2)


class TestBaselineValidation(unittest.TestCase):
    def test_malformed_baselines_rejected(self):
        from Deployement.normalizer import validate_baselines
        bad_cases = [
            ["not-a-dict"],
            {"processes": "sshd"},                       # bare string
            {"processes": ["sshd", 22]},                 # non-string entry
            {"processes": [""]},                         # blank entry
            {"ports": ["http"]},                         # non-numeric port
            {"ports": [True]},                           # bool is not a port
            {"ports": [-1]},                             # out of range
            {"ports": [65536]},                          # out of range
            {"peers": "10.0.0.1"},                       # bare string
            {"peers": ["ok", ""]},                       # blank entry
            {"bogus_dimension": []},                     # unknown field
            {"roles": ["server"]},                       # scopes not a dict
            {"roles": {"server": ["sshd"]}},             # scope not a dict
            {"roles": {"server": {"bogus": []}}},        # unknown field
            {"roles": {"": {"processes": []}}},          # blank scope name
            {"zones": {"z": {"ports": [22, "abc"]}}},    # bad port in scope
            {"hosts": {"h": {"processes": "x"}}},        # bad scope shape
        ]
        for bad in bad_cases:
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_baselines(bad)

    def test_valid_baselines_accepted(self):
        from Deployement.normalizer import validate_baselines
        good = {"processes": ["sshd"], "ports": [22, "443"],
                "peers": {"ctrl"},
                "roles": {"server": {"ports": [22]}},
                "zones": {}, "hosts": {}}
        self.assertEqual(validate_baselines(good), good)
        self.assertEqual(validate_baselines(None), {})
        # Constructor enforces the same rules up front.
        from Deployement.normalizer import TelemetryNormalizer
        with self.assertRaises(ValueError):
            TelemetryNormalizer({"ports": ["http"]})


class TestPeerMatching(unittest.TestCase):
    def test_adversarial_prefixes(self):
        from Deployement.normalizer import peer_is_known
        # The exact substring trap the old code had:
        self.assertFalse(peer_is_known("tcp:22->host10", {"host1"}))
        self.assertFalse(peer_is_known("tcp:22->host1-extra", {"host1"}))
        self.assertFalse(peer_is_known("tcp:22->10.0.0.10", {"10.0.0.1"}))
        self.assertFalse(peer_is_known("tcp:22->10.0.0.100", {"10.0.0.10"}))

    def test_legitimate_matches_kept(self):
        from Deployement.normalizer import peer_is_known
        self.assertTrue(peer_is_known("tcp:22->10.0.0.1", {"10.0.0.1"}))
        self.assertTrue(peer_is_known("TCP:22->HOST1", {"host1"}))
        self.assertTrue(peer_is_known("tcp:22->host1.", {"host1"}))
        self.assertTrue(peer_is_known("tcp:22->10.0.0.5", {"10.0.0"}))
        self.assertTrue(peer_is_known("tcp:22->10.0.0.5:443", {"10.0.0.5"}))
        self.assertFalse(peer_is_known("tcp:22", {"10.0.0.1"}))
        self.assertFalse(peer_is_known("", {"10.0.0.1"}))
        self.assertFalse(peer_is_known("tcp:22->", {"10.0.0.1"}))

    def test_hostname_prefixes_never_match(self):
        # Distinct machines must never collapse into one identity, even
        # with dash/underscore/dot separators.
        from Deployement.normalizer import peer_is_known
        for peer in ("host1-extra", "host1_backup", "host1.db", "host10",
                     "host11"):
            self.assertFalse(peer_is_known(f"tcp:22->{peer}", {"host1"}),
                             msg=peer)
        self.assertFalse(peer_is_known("tcp:22->db-01", {"db"}))
        self.assertFalse(peer_is_known("tcp:22->db_01", {"db"}))


class TestPortParsing(unittest.TestCase):
    def test_valid_records(self):
        from Deployement.normalizer import _port_of
        self.assertEqual(_port_of("tcp:22->10.0.0.1"), 22)
        self.assertEqual(_port_of("udp:53->dns"), 53)
        self.assertEqual(_port_of("tcp:443"), 443)
        self.assertEqual(_port_of("TCP:80->h"), 80)
        self.assertEqual(_port_of("icmp:0->h"), 0)
        self.assertEqual(_port_of("tcp:65535->h"), 65535)

    def test_malformed_records_rejected(self):
        from Deployement.normalizer import _port_of
        for bad in ("tcp:22x->h", "tcp:99999->h", "tcp:65536->h",
                    "tcp:->h", "tcp->h", "22", "", "   ",
                    "tcp:22:33->h", ":22->h", "1:80->h", "tcp: 22->h",
                    None, 22, ["tcp:22"]):
            self.assertIsNone(_port_of(bad), msg=repr(bad))


class TestQueuedFlag(unittest.TestCase):
    def test_queue_paths_set_queued_flag(self):
        from Deployement.executor import EnforcementState, MockBackend
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        tables = build_all_tables(amap)
        restore0 = next(i for i, e in enumerate(tables[0])
                        if e["command"] == "Restore")
        for mode in ("supervised",):
            pipeline = DeploymentPipeline(
                amap, demo_collector(amap),
                _FixedPolicy([restore0, 16, 16, 16, 48]), mode=mode,
                baselines=demo_baselines(), cooldown_s=0)
            pipeline.step()  # quiet: masked -> rejected
            pipeline.step()  # scan alerts: queued
            self.assertTrue(pipeline.enforcement.pending_approvals)
            # Direct executor-level flag check on the queued path.
            from Deployement.executor import SupervisedExecutor
            state = EnforcementState()
            executor = SupervisedExecutor(state)
            result = executor.execute(
                {"operation": "reimage_host", "target": "h",
                 "needs_approval": True, "approved": False})
            self.assertTrue(result.queued)
            self.assertFalse(result.applied)
            pipeline.close()

    def test_live_queue_path_sets_queued_flag(self):
        from Deployement.executor import EnforcementState, LiveExecutor
        state = EnforcementState()
        executor = LiveExecutor(state)
        result = executor.execute(
            {"operation": "reimage_host", "zone": "z", "target": "h",
             "needs_approval": True, "approved": False,
             "agent_id": 0, "command": "Restore",
             "idempotency_key": "q-flag"})
        self.assertTrue(result.queued)
        self.assertFalse(result.applied)


class TestHistoryAndRotation(unittest.TestCase):
    def test_history_trimmed(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)), mode="shadow",
            baselines=demo_baselines(), history_max=7)
        pipeline.run()
        pipeline.close()
        self.assertLessEqual(len(pipeline.history), 7)
        self.assertEqual(pipeline.cycle, 5)

    def test_log_rotation(self):
        import json
        import os
        import tempfile
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DeploymentPipeline(
                amap, demo_collector(amap),
                MockPolicyEngine(build_all_tables(amap)), mode="shadow",
                baselines=demo_baselines(), session_dir=tmp,
                max_log_bytes=1)
            pipeline.run()
            pipeline.close()
            main_log = os.path.join(tmp, "decisions.jsonl")
            backup = main_log + ".1"
            # Rotation happened and logging continued afterwards; the
            # live file always ends with the latest records.
            self.assertTrue(os.path.exists(backup))
            self.assertTrue(os.path.exists(main_log))

            def read(path):
                with open(path, encoding="utf-8") as fh:
                    return [json.loads(line) for line in fh
                            if line.strip()]

            main_rows, backup_rows = read(main_log), read(backup)
            self.assertTrue(main_rows)
            self.assertTrue(backup_rows)
            # No record lost across the rotation boundary.
            cycles = sorted(r["cycle"] for r in main_rows + backup_rows)
            self.assertEqual(cycles[0], min(cycles))
            self.assertEqual(len(main_rows) + len(backup_rows), 10)
            self.assertEqual(max(cycles), 4)


def EndpointConfig_for_test(url, **over):
    from Deployement.live import EndpointConfig
    params = {"name": "test", "endpoint": url, "timeout_s": 5.0}
    params.update(over)
    return EndpointConfig(**params)


class TestRedirectRefused(unittest.TestCase):
    def test_redirect_is_loud_failure_not_follow(self):
        from http.server import BaseHTTPRequestHandler
        from Deployement.live import CollectorError, HttpJsonCollector

        class Redirector(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:9/evil")
                self.end_headers()

            def log_message(self, *args):
                pass

        with _LocalHTTPServer(Redirector) as server:
            collector = HttpJsonCollector(
                EndpointConfig_for_test(server.url), _translate_ok)
            with self.assertRaises(CollectorError) as ctx:
                collector.next_batch()
        self.assertIn("redirect", str(ctx.exception).lower())


class TestDeviceParam(unittest.TestCase):
    def test_matching_device_accepted(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule(), tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "good.pt")
            torch.save({"model": {}, "num_host_targets": 137,
                        "num_subnet_targets": 9}, path)
            mappo, meta = model_loader.load_trained_mappo(
                path, device="cpu")
            self.assertEqual(meta["device"], "cpu")

    def test_mismatched_device_rejected_loudly(self):
        import torch
        from Deployement import model_loader
        with _FakeMarlModule(), tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "good.pt")
            torch.save({"model": {}, "num_host_targets": 137,
                        "num_subnet_targets": 9}, path)
            with self.assertRaises(model_loader.DeploymentError) as ctx:
                model_loader.load_trained_mappo(path, device="cuda")
            self.assertIn("cuda", str(ctx.exception))


class TestMockPolicyFallback(unittest.TestCase):
    def test_all_false_mask_falls_back_without_crash(self):
        import numpy as np
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        tables = build_all_tables(amap)
        engine = MockPolicyEngine(tables)
        obs = np.zeros((5, 210), dtype=np.float32)
        masks = [np.zeros(len(tables[i]), dtype=bool) for i in range(5)]
        decisions = engine.decide(obs, masks)
        # Every decision resolves to a real table entry (Sleep fallback).
        for agent, record in enumerate(decisions):
            self.assertIn(record["action"], range(len(tables[agent])))

    def test_latency_present_and_numeric(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)), mode="shadow",
            baselines=demo_baselines())
        records = pipeline.step()
        pipeline.close()
        for record in records:
            self.assertIsInstance(record["latency_ms"], float)
            self.assertGreaterEqual(record["latency_ms"], 0.0)


class _StubObjects:
    """Tiny attribute bags for headless model tests (no Tk needed)."""

    @staticmethod
    def pipeline(hosts=None, history=None, health="ok", pending=None,
                 policy_error="", unseen=None, asset_map=None,
                 enforcement_audit=None):
        import types
        return types.SimpleNamespace(
            last_normalized=types.SimpleNamespace(hosts=hosts or {}),
            history=list(history or []),
            health=types.SimpleNamespace(state=health),
            enforcement=types.SimpleNamespace(
                pending_approvals=list(pending or []),
                audit=list(enforcement_audit or [])),
            last_policy_error=policy_error,
            last_unseen=list(unseen or []),
            asset_map=asset_map,
            last_batch=None, last_obs=None, last_masks=None,
            tables={}, policy=None, policy_info={},
            session_id="testsession", cycle=0,
            state_builder=types.SimpleNamespace(mission_phase=0),
            validator=types.SimpleNamespace(mode="shadow"))

    @staticmethod
    def host(cc4, severity="none", health="quiet", compromised=False,
             notes=(), up=True, sessions=0):
        import types
        return types.SimpleNamespace(
            cc4=cc4, key="k", compromised=compromised,
            process_event=False, connection_event=False,
            session_count=sessions, up=up, health=health,
            last_seen=0.0, severity=severity, notes=list(notes))


class TestGuiModels(unittest.TestCase):
    """Headless tests for every dashboard model function."""

    def test_overview_idle_no_data(self):
        from Deployement import gui
        status = gui.overall_status(_StubObjects.pipeline())
        self.assertEqual(status["level"], "idle")
        self.assertIn("Step", status["headline"])

    def test_overview_critical_and_attention(self):
        from Deployement import gui
        bad = _StubObjects.pipeline(
            hosts={"h1": _StubObjects.host("h1", severity="critical",
                                           health="compromised",
                                           compromised=True)})
        status = gui.overall_status(bad)
        self.assertEqual(status["level"], "critical")
        self.assertIn("h1", status["reasons"][0])
        stale = _StubObjects.pipeline(
            hosts={"h1": _StubObjects.host("h1", severity="unknown",
                                           health="stale")})
        self.assertEqual(gui.overall_status(stale)["level"], "attention")
        down = _StubObjects.pipeline()
        down.health.state = "down"
        self.assertEqual(gui.overall_status(down)["level"], "critical")

    def test_threat_list_filter_search(self):
        from Deployement import gui
        from Deployement.asset_map import generate_default_map
        amap = generate_default_map()
        hosts = {
            "restricted_zone_a_subnet_server_host_0": _StubObjects.host(
                "restricted_zone_a_subnet_server_host_0",
                severity="critical", health="compromised",
                compromised=True, notes=["intrusion evidence"]),
            "restricted_zone_a_subnet_server_host_1": _StubObjects.host(
                "restricted_zone_a_subnet_server_host_1",
                severity="high", health="quiet"),
            "restricted_zone_a_subnet_user_host_0": _StubObjects.host(
                "restricted_zone_a_subnet_user_host_0",
                severity="low", health="quiet"),
        }
        pipeline = _StubObjects.pipeline(hosts=hosts, asset_map=amap)
        rows = gui.threat_list(pipeline)
        self.assertEqual([r["severity"] for r in rows],
                         ["critical", "high", "low"])
        self.assertEqual(len(gui.threat_list(pipeline, severity="high")), 1)
        ip = amap.agents["blue_agent_0"]["hosts"][
            "restricted_zone_a_subnet_server_host_0"]["ip"]
        found = gui.threat_list(pipeline, query=ip)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["status"], "active")
        self.assertEqual(gui.threat_list(pipeline, query="zzz-no-match"), [])

    def test_threat_detail_timeline(self):
        from Deployement import gui
        from Deployement.asset_map import generate_default_map
        amap = generate_default_map()
        cc4 = "restricted_zone_a_subnet_server_host_0"
        record = {"cycle": 2, "timestamp": 3.0, "agent_id": 0,
                  "label": f"Restore {cc4}", "status": "queued",
                  "operation": "reimage_host", "risk": "destructive"}
        pipeline = _StubObjects.pipeline(
            hosts={cc4: _StubObjects.host(
                cc4, severity="critical", health="compromised",
                compromised=True, notes=["intrusion evidence"])},
            history=[record], asset_map=amap,
            enforcement_audit=[{"event": "queued", "operation": "Restore",
                                "target": cc4}])
        detail = gui.threat_detail(pipeline, cc4)
        self.assertEqual(detail["host"]["cc4"], cc4)
        self.assertEqual(len(detail["timeline"]), 1)
        self.assertEqual(detail["timeline"][0]["label"], f"Restore {cc4}")
        self.assertEqual(len(detail["audit"]), 1)
        self.assertIsNone(gui.threat_detail(pipeline, "nope")["host"])

    def test_agent_rows(self):
        from Deployement import gui
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)),
            mode="shadow", baselines=demo_baselines())
        try:
            pipeline.step()
            rows = gui.agent_rows(pipeline)
            self.assertEqual(len(rows), 5)
            by_id = {r["agent_id"]: r for r in rows}
            self.assertEqual(by_id[0]["zones"],
                             ["restricted_zone_a_subnet"])
            self.assertEqual(by_id[0]["host_count"], 16)
            self.assertEqual(by_id[4]["host_count"], 48)
            for row in rows:
                self.assertIn("health", row)
                self.assertIn("confidence", row)
                self.assertIn("comm", row)
        finally:
            pipeline.close()

    def test_action_rows_human_readable(self):
        from Deployement import gui
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)),
            mode="shadow", baselines=demo_baselines())
        try:
            pipeline.step()
            rows = gui.action_rows(pipeline)
            self.assertEqual(len(rows), 5)
            verbs = set(gui._HUMAN_VERBS.values())
            for row in rows:
                self.assertIn(row["action"], verbs)
                self.assertTrue(row["target"])
                self.assertIn(row["risk"], ("safe", "elevated",
                                            "destructive", "?"))
        finally:
            pipeline.close()

    def test_notification_list(self):
        from Deployement import gui
        pipeline = _StubObjects.pipeline(
            hosts={"h1": _StubObjects.host("h1", severity="critical",
                                           health="compromised",
                                           compromised=True)},
            pending=[{"operation": "Restore", "target": "h1"}],
            health="ok")
        pipeline.health.state = "down"
        items = gui.notification_list(pipeline)
        self.assertGreaterEqual(len(items), 3)
        self.assertEqual(items[0]["level"], "critical")
        kinds = {item["kind"] for item in items}
        self.assertTrue({"threat", "approvals", "telemetry"} <= kinds)
        self.assertEqual(gui.notification_list(
            _StubObjects.pipeline()), [])

    def test_search_all_kinds(self):
        from Deployement import gui
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)),
            mode="shadow", baselines=demo_baselines())
        try:
            pipeline.step()
            pipeline.step()
            pipeline.step()
            pipeline.enforcement.audit.append(
                {"event": "executed", "operation": "reimage_host",
                 "target": "somehost", "mode": "mock", "applied": True,
                 "details": "", "error": ""})
            ip = amap.agents["blue_agent_0"]["hosts"][
                "restricted_zone_a_subnet_server_host_0"]["ip"]
            self.assertTrue(any(r["kind"] == "host"
                                for r in gui.search(pipeline, ip)))
            self.assertTrue(any(r["kind"] == "agent"
                                for r in gui.search(pipeline, "blue_agent_1")))  # noqa: E501
            self.assertTrue(any(r["kind"] == "audit"
                                for r in gui.search(pipeline, "reimage")))
            self.assertTrue(any(r["kind"] == "tab"
                                for r in gui.search(pipeline, "model")))
            self.assertEqual(gui.search(pipeline, ""), [])
            self.assertEqual(gui.search(pipeline, "   "), [])
        finally:
            pipeline.close()

    def test_telemetry_model(self):
        from Deployement import gui
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)),
            mode="shadow", baselines=demo_baselines())
        try:
            pipeline.step()
            model = gui.telemetry_model(pipeline)
            for key in ("state", "total_batches", "last_update_ts",
                        "events_last_batch", "partial_errors",
                        "stale_hosts", "degraded"):
                self.assertIn(key, model)
            self.assertEqual(model["total_batches"], 1)
            self.assertGreater(model["events_last_batch"], 0)
            rate = gui.telemetry_model(
                pipeline, event_log=[(1000.0, 10), (1010.0, 20)],
                now=1015.0)["events_per_sec"]
            self.assertAlmostEqual(rate, 3.0)  # 30 events / 10 s span
            self.assertIsNone(
                gui.telemetry_model(pipeline,
                                    event_log=[(1000.0, 5)])
                ["events_per_sec"])
        finally:
            pipeline.close()

    def test_model_page_model(self):
        from Deployement import gui
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)), mode="shadow",
            baselines=demo_baselines(),
            policy_info={"engine": "mock-policy", "checkpoint": None})
        try:
            model = gui.model_page_model(pipeline)
            self.assertEqual(model["engine"], "mock-policy")
            self.assertFalse(model["checkpoint_loaded"])
            self.assertTrue(model["vocab_match"])
            self.assertEqual(model["vocab_host"], 137)
            self.assertEqual(model["action_sizes"],
                             {0: 82, 1: 82, 2: 82, 3: 82, 4: 242})
            self.assertIsNone(model["latency_avg_ms"])
            pipeline.step()
            model = gui.model_page_model(pipeline)
            self.assertIsInstance(model["latency_avg_ms"], float)
            # last_obs is the padded [5, 210] model batch.
            self.assertEqual(model["obs_rows"], [210] * 5)
        finally:
            pipeline.close()

    def test_audit_rows_and_export(self):
        import json
        import os
        import tempfile
        from Deployement import gui
        entries = [
            {"event": "queued", "operation": "Restore", "target": "h1",
             "mode": "supervised", "applied": False},
            {"event": "executed", "operation": "Monitor", "target": "",
             "mode": "shadow", "applied": False},
            "junk-string-entry",
            {"event": "failed", "operation": "reimage_host",
             "target": "h2", "mode": "live", "applied": False,
             "error": "boom"},
        ]
        self.assertEqual(len(gui.audit_rows(entries)), 3)
        self.assertEqual(len(gui.audit_rows(entries, event="failed")), 1)
        self.assertEqual(len(gui.audit_rows(entries, query="h1")), 1)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "audit.csv")
            gui.export_audit(entries, csv_path, format="csv")
            with open(csv_path, encoding="utf-8") as handle:
                header = handle.readline()
            self.assertIn("operation", header)
            json_path = os.path.join(tmp, "audit.json")
            gui.export_audit(entries, json_path, format="json")
            with open(json_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertEqual(len(loaded), 3)
            with self.assertRaises(ValueError):
                gui.export_audit(entries, os.path.join(tmp, "x.xml"),
                                 format="xml")

    def test_topology_layout_and_hit(self):
        from Deployement import gui
        layout = gui.topology_layout()
        self.assertEqual(len(layout["subnets"]), 9)
        self.assertEqual(len(layout["agents"]), 5)
        x, y = layout["subnets"]["restricted_zone_a_subnet"]
        self.assertEqual(gui.topology_hit(layout, x, y),
                         ("subnet", "restricted_zone_a_subnet"))
        ax, ay = layout["agents"][2]
        self.assertEqual(gui.topology_hit(layout, ax, ay), ("agent", 2))
        self.assertIsNone(gui.topology_hit(layout, -10, -10))

    def test_confidence_formatting(self):
        from Deployement import gui
        self.assertEqual(gui.confidence_of(None), "—")
        self.assertEqual(gui.confidence_of({}), "—")
        self.assertEqual(gui.confidence_of({"confidence": 0.0}), "—")
        self.assertEqual(gui.confidence_of({"confidence": 0.82}), "82%")


class TestGuiDashboard(unittest.TestCase):
    """Widget-level dashboard tests (need a display; skipped headless)."""

    def _root(self):
        try:
            import tkinter
        except Exception:
            self.skipTest("tkinter unavailable")
        try:
            root = tkinter.Tk()
            root.withdraw()
            self.addCleanup(root.destroy)
            return root
        except Exception as exc:
            self.skipTest(f"no display: {exc}")

    def _stepped_pipeline(self, mode="shadow", cycles=3):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.policy_engine import MockPolicyEngine
        amap = generate_default_map()
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            MockPolicyEngine(build_all_tables(amap)), mode=mode,
            baselines=demo_baselines())
        for _ in range(cycles):
            pipeline.step()
        self.addCleanup(pipeline.close)
        return pipeline

    def _console(self, root, pipeline=None):
        from Deployement import gui
        return gui.DeploymentConsole(
            root, lambda: (pipeline, {"policy": "mock-policy",
                                      "assets": "default"}))

    def test_all_pages_show(self):
        from Deployement import gui
        root = self._root()
        console = self._console(root)
        try:
            for page in gui.PAGES:
                console.show_page(page)
                self.assertEqual(console._current_page, page)
            console.show_page("Nope")
            self.assertEqual(console._current_page, "Settings")
        finally:
            root.update_idletasks()

    def test_topology_click_selects(self):
        from Deployement import gui
        import types
        root = self._root()
        pipeline = self._stepped_pipeline()
        console = self._console(root)
        try:
            console.pipeline = pipeline
            console.show_page("Network")
            layout = gui.topology_layout()
            x, y = layout["subnets"]["restricted_zone_a_subnet"]
            console._on_topo_click(types.SimpleNamespace(x=x, y=y))
            self.assertEqual(console._topo_selection,
                             ("subnet", "restricted_zone_a_subnet"))
            self.assertTrue(
                console._topo_hosts.size() > 0)
            ax, ay = layout["agents"][4]
            console._on_topo_click(types.SimpleNamespace(x=ax, y=ay))
            self.assertEqual(console._topo_selection, ("agent", 4))
        finally:
            root.update_idletasks()

    def test_host_modal_open_close(self):
        from Deployement import gui
        root = self._root()
        pipeline = self._stepped_pipeline()
        console = self._console(root)
        try:
            console.pipeline = pipeline
            cc4 = "restricted_zone_a_subnet_server_host_0"
            console._open_host(cc4)
            self.assertEqual(len(console._modals), 1)
            console._close_top_modal()
            self.assertEqual(len(console._modals), 0)
            before = len(console._modals)
            console._open_host("no-such-host")
            self.assertEqual(len(console._modals), before)
        finally:
            root.update_idletasks()

    def test_action_approve_deny(self):
        from Deployement.action_table import build_all_tables
        from Deployement.asset_map import generate_default_map
        from Deployement.demo import demo_baselines, demo_collector
        from Deployement.pipeline import DeploymentPipeline
        from Deployement.tests.test_deployment import _FixedPolicy
        root = self._root()
        amap = generate_default_map()
        tables = build_all_tables(amap)
        restore0 = next(i for i, e in enumerate(tables[0])
                        if e["command"] == "Restore")
        pipeline = DeploymentPipeline(
            amap, demo_collector(amap),
            _FixedPolicy([restore0, 16, 16, 16, 48]), mode="supervised",
            baselines=demo_baselines(), cooldown_s=0)
        self.addCleanup(pipeline.close)
        # Cycles 0-2: quiet (masked), scan (queued), compromise (queued).
        pipeline.step()
        pipeline.step()
        pipeline.step()
        self.assertEqual(len(pipeline.enforcement.pending_approvals), 2)
        from Deployement import gui
        console = self._console(root)
        try:
            console.pipeline = pipeline
            console.mode_var.set("supervised")
            console.show_page("Actions")
            tree = console._tab_approvals.tree
            children = tree.get_children()
            self.assertEqual(len(children), 2)
            tree.selection_set(children[0])
            # Placeholder rows are never valid selections.
            self.assertIsNotNone(console._selected_approval())
            console._approve_selected()
            self.assertEqual(len(pipeline.enforcement.pending_approvals), 1)
            children = tree.get_children()
            self.assertEqual(len(children), 1)
            tree.selection_set(children[0])
            self.assertIsNotNone(console._selected_approval())
            console._deny_selected()
            self.assertEqual(len(pipeline.enforcement.pending_approvals), 0)
        finally:
            root.update_idletasks()

    def test_search_and_notifications_modals(self):
        from Deployement import gui
        root = self._root()
        pipeline = self._stepped_pipeline()
        console = self._console(root)
        try:
            console.pipeline = pipeline
            console.search_var.set("blue_agent")
            console._open_search()
            self.assertEqual(len(console._modals), 1)
            console._close_top_modal()
            console._open_notifications()
            self.assertEqual(len(console._modals), 1)
            console._close_top_modal()
            self.assertEqual(len(console._modals), 0)
        finally:
            root.update_idletasks()

    def test_action_details_modal(self):
        from Deployement import gui
        root = self._root()
        pipeline = self._stepped_pipeline()
        console = self._console(root)
        try:
            console.pipeline = pipeline
            console.show_page("Actions")
            rows = gui.action_rows(pipeline)
            self.assertTrue(rows)
            console._open_action(rows[0]["record"])
            self.assertEqual(len(console._modals), 1)
            console._close_top_modal()
        finally:
            root.update_idletasks()

    def test_theme_toggle_and_keyboard(self):
        from Deployement import gui
        root = self._root()
        console = self._console(root)
        try:
            self.assertEqual(console._theme, "dark")
            palette = gui.apply_theme(root, "dark")
            self.assertIn("page_bg", palette)
            console._toggle_theme()
            self.assertEqual(console._theme, "light")
            console._toggle_theme()
            self.assertEqual(console._theme, "dark")
            # Keyboard wiring: Alt+1..9 bound per page, Escape closes.
            for index in range(1, 10):
                self.assertTrue(root.bind(f"<Alt-KeyPress-{index}>"),
                                f"Alt+{index} not bound")
            self.assertTrue(root.bind("<Escape>"))
            with self.assertRaises(ValueError):
                gui.apply_theme(root, "neon")
        finally:
            root.update_idletasks()

    def test_mode_banner_updates(self):
        from Deployement import gui
        root = self._root()
        console = self._console(root)
        try:
            console.mode_var.set("supervised")
            console._on_mode_change()
            self.assertIn("SUPERVISED", console.banner_mode_var.get())
        finally:
            root.update_idletasks()


if __name__ == "__main__":
    unittest.main()
