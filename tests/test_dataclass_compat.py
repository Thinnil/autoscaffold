"""
tests/test_dataclass_compat.py
Comprehensive unit tests for zero-dependency dataclass compatibility shim.

Validates:
1. Native and fallback dataclass implementations.
2. Default literals and freshness of default_factory (lists, dicts).
3. Parameter ordering enforcement (non-default after default).
4. Class properties, methods, and descriptors preservation.
5. Frozen instance protection and FrozenInstanceError.
6. Recursive asdict() serialization and is_dataclass() introspection.
7. Inheritance across MRO hierarchy.
8. Clean imports of all AutoScaffold stages under simulated bare Python 3.6.
"""

import importlib
import os
import sys
import unittest
import numpy as np

# Resolve repo root and scripts path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

for p in (SCRIPTS_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import scripts.dataclass_compat as dc  # noqa: E402


class TestDataclassCompatFallback(unittest.TestCase):
    """Direct validation of the pure-Python fallback implementation."""

    def test_basic_fallback_dataclass(self):
        @dc._FallbackDataclass
        class Point:
            x: int
            y: int

        p1 = Point(10, 20)
        self.assertEqual(p1.x, 10)
        self.assertEqual(p1.y, 20)
        self.assertEqual(repr(p1), "Point(x=10, y=20)")
        self.assertTrue(dc._fallback_is_dataclass(Point))
        self.assertTrue(dc._fallback_is_dataclass(p1))

        p2 = Point(x=10, y=20)
        self.assertEqual(p1, p2)

        p3 = Point(10, 25)
        self.assertNotEqual(p1, p3)
        self.assertNotEqual(p1, (10, 20))

    def test_default_values_and_default_factory_freshness(self):
        @dc._FallbackDataclass
        class Container:
            title: str
            tags: list = dc._fallback_field(default_factory=list)
            metadata: dict = dc._fallback_field(default_factory=dict)
            priority: int = 5

        c1 = Container("Alpha")
        c2 = Container("Beta")

        self.assertEqual(c1.title, "Alpha")
        self.assertEqual(c1.priority, 5)
        self.assertEqual(c1.tags, [])
        self.assertEqual(c1.metadata, {})

        # Mutate c1 collections: must NOT affect c2
        c1.tags.append("urgent")
        c1.metadata["retries"] = 3

        self.assertEqual(c1.tags, ["urgent"])
        self.assertEqual(c1.metadata, {"retries": 3})
        self.assertEqual(c2.tags, [])
        self.assertEqual(c2.metadata, {})
        self.assertIsNot(c1.tags, c2.tags)
        self.assertIsNot(c1.metadata, c2.metadata)

    def test_non_default_after_default_raises_type_error(self):
        with self.assertRaises(TypeError):
            @dc._FallbackDataclass
            class InvalidOrder:
                x: int = 10
                y: int

    def test_both_default_and_factory_raises_value_error(self):
        with self.assertRaises(ValueError):
            dc._fallback_field(default=10, default_factory=list)

    def test_properties_and_methods_preserved(self):
        @dc._FallbackDataclass
        class ResidueInfo:
            res_name: str
            res_id: int

            @property
            def identifier(self) -> str:
                return f"{self.res_name}_{self.res_id}"

            def is_glycine(self) -> bool:
                return self.res_name == "GLY"

        r = ResidueInfo("ALA", 42)
        self.assertEqual(r.identifier, "ALA_42")
        self.assertFalse(r.is_glycine())

        r_gly = ResidueInfo("GLY", 10)
        self.assertTrue(r_gly.is_glycine())

        # Verify identifier is not included in dataclass fields
        fields = getattr(ResidueInfo, "__dataclass_fields__")
        self.assertIn("res_name", fields)
        self.assertIn("res_id", fields)
        self.assertNotIn("identifier", fields)
        self.assertNotIn("is_glycine", fields)

    def test_frozen_instance_raises_frozen_instance_error(self):
        @dc._FallbackDataclass(frozen=True)
        class ImmutableCoord:
            x: float
            y: float
            label: str = "origin"

        pt = ImmutableCoord(1.5, 2.5)
        self.assertEqual(pt.x, 1.5)
        self.assertEqual(pt.y, 2.5)
        self.assertEqual(pt.label, "origin")

        err_types = (dc._FallbackFrozenInstanceError, AttributeError)
        with self.assertRaises(err_types):
            pt.x = 9.9

        with self.assertRaises(err_types):
            del pt.label

    def test_recursive_asdict(self):
        @dc._FallbackDataclass
        class Node:
            id: int
            name: str

        @dc._FallbackDataclass
        class Cluster:
            root: Node
            nodes: list
            mapping: dict

        tree = Cluster(
            root=Node(1, "master"),
            nodes=[Node(2, "worker-1"), Node(3, "worker-2")],
            mapping={"active": Node(2, "worker-1")},
        )

        d = dc._fallback_asdict(tree)
        expected = {
            "root": {"id": 1, "name": "master"},
            "nodes": [
                {"id": 2, "name": "worker-1"},
                {"id": 3, "name": "worker-2"},
            ],
            "mapping": {
                "active": {"id": 2, "name": "worker-1"},
            },
        }
        self.assertEqual(d, expected)

        # Calling asdict on non-dataclass raises TypeError
        with self.assertRaises(TypeError):
            dc._fallback_asdict("not a dataclass")

    def test_inheritance_mro_ordering(self):
        @dc._FallbackDataclass
        class Parent:
            a: int
            b: str = "default_b"

        @dc._FallbackDataclass
        class Child(Parent):
            c: float = 3.14

        ch = Child(10)
        self.assertEqual(ch.a, 10)
        self.assertEqual(ch.b, "default_b")
        self.assertEqual(ch.c, 3.14)

        ch2 = Child(20, "custom_b", 9.99)
        self.assertEqual(ch2.a, 20)
        self.assertEqual(ch2.b, "custom_b")
        self.assertEqual(ch2.c, 9.99)


class TestSimulatedPython36Environment(unittest.TestCase):
    """Simulates bare Python 3.6 by masking stdlib 'dataclasses'."""

    def test_simulated_python36_and_autoscaffold_stages_import(self):
        orig_dataclasses = sys.modules.get("dataclasses")
        try:
            # Mask native dataclasses module to simulate bare Python 3.6
            sys.modules["dataclasses"] = None

            # Reload dataclass_compat in simulated mode
            importlib.reload(dc)
            self.assertFalse(dc._HAS_NATIVE_DATACLASSES)
            self.assertIs(dc.dataclass, dc._FallbackDataclass)
            self.assertIs(dc.field, dc._fallback_field)
            self.assertIs(dc.asdict, dc._fallback_asdict)
            self.assertIs(dc.is_dataclass, dc._fallback_is_dataclass)
            self.assertIs(
                dc.FrozenInstanceError, dc._FallbackFrozenInstanceError
            )

            # 1. Test Stage 1: MotifAtom, MotifResidue, MotifSegment
            import scripts.chimera_stages.stage1_parsing as s1
            importlib.reload(s1)

            atom = s1.MotifAtom(
                atom_name="CA",
                element="C",
                coords=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            )
            self.assertEqual(atom.atom_name, "CA")
            self.assertEqual(atom.occupancy, 1.0)
            self.assertEqual(atom.b_factor, 0.0)
            # Verify frozen protection
            with self.assertRaises((dc.FrozenInstanceError, AttributeError)):
                atom.occupancy = 0.5

            res = s1.MotifResidue(
                source_file="donor.pdb",
                source_chain="A",
                orig_resi=10,
                resn="ALA",
            )
            self.assertEqual(res.atoms, {})
            self.assertEqual(res.insertion_code, "")
            self.assertIsNone(res.ca_coords)

            # 2. Test Stage 4: RunnerSpec
            import scripts.chimera_stages.stage4_runners as s4
            importlib.reload(s4)

            runner = s4.RunnerSpec(
                runner_type="rfdiffusion",
                command_prefix=["micromamba", "run", "-n", "SE3nv"],
            )
            self.assertEqual(runner.runner_type, "rfdiffusion")
            self.assertEqual(runner.missing_modules, [])
            self.assertFalse(runner.is_valid)

            # 3. Test Central Orchestrator: PipelineContext, PipelineResult
            import scripts.run_chimera_pipeline as rcp
            importlib.reload(rcp)

            ctx = rcp.PipelineContext()
            self.assertEqual(ctx.preserved_donor_segments, [])
            self.assertEqual(ctx.donor_metadata, {})

            res = rcp.PipelineResult(status="SUCCESS")
            self.assertEqual(res.status, "SUCCESS")
            self.assertEqual(res.summary_data, {})
            self.assertEqual(res.passing_models, [])

            # 4. Test Stage Init & Chimera Scaffold facade
            import scripts.chimera_stages as cs_stages
            importlib.reload(cs_stages)
            self.assertTrue(hasattr(cs_stages, "dataclass"))
            self.assertTrue(hasattr(cs_stages, "MotifAtom"))

            import scripts.chimera_scaffold as cs
            importlib.reload(cs)
            self.assertTrue(hasattr(cs, "dataclass"))
            self.assertTrue(hasattr(cs, "MotifAtom"))
            self.assertTrue(hasattr(cs, "RunnerSpec"))

        finally:
            # Restore original environment
            if orig_dataclasses is not None:
                sys.modules["dataclasses"] = orig_dataclasses
            else:
                sys.modules.pop("dataclasses", None)

            importlib.reload(dc)
            import scripts.chimera_stages.stage1_parsing as s1
            importlib.reload(s1)
            import scripts.chimera_stages.stage4_runners as s4
            importlib.reload(s4)
            import scripts.run_chimera_pipeline as rcp
            importlib.reload(rcp)
            import scripts.chimera_stages as cs_stages
            importlib.reload(cs_stages)
            import scripts.chimera_scaffold as cs
            importlib.reload(cs)


class TestDataclassCompatNative(unittest.TestCase):
    """Validation of zero-overhead native passthrough on Python >= 3.7."""

    def test_native_passthrough_identity(self):
        if sys.version_info >= (3, 7):
            import dataclasses as std_dc

            self.assertTrue(dc._HAS_NATIVE_DATACLASSES)
            self.assertIs(dc.dataclass, std_dc.dataclass)
            self.assertIs(dc.field, std_dc.field)
            self.assertIs(dc.asdict, std_dc.asdict)
            self.assertIs(dc.is_dataclass, std_dc.is_dataclass)
            self.assertIs(dc.FrozenInstanceError, std_dc.FrozenInstanceError)
            self.assertIs(dc.Field, std_dc.Field)
            self.assertIs(dc.MISSING, std_dc.MISSING)


class TestDataclassCompatOptimizations(unittest.TestCase):
    """Validation of micro-optimizations in fallback classes."""

    def test_fallback_field_slots_and_copy(self):
        f = dc._fallback_field(default=42, compare=False)
        self.assertTrue(hasattr(f, "__slots__"))
        self.assertFalse(hasattr(f, "__dict__"))
        self.assertEqual(f.default, 42)
        self.assertFalse(f.compare)

        # Verify copy
        import copy
        f_copy = copy.copy(f)
        self.assertEqual(f_copy.default, 42)
        self.assertFalse(f_copy.compare)

    def test_fallback_asdict_atomic_passthrough_and_speed(self):
        @dc._FallbackDataclass
        class HeavyRecord:
            id: int
            name: str
            ratio: float
            flag: bool
            payload: list

        rec = HeavyRecord(101, "sample", 3.14159, True, [1, 2, 3])
        d = dc._fallback_asdict(rec)
        self.assertEqual(d["id"], 101)
        self.assertEqual(d["name"], "sample")
        self.assertEqual(d["ratio"], 3.14159)
        self.assertEqual(d["flag"], True)
        self.assertEqual(d["payload"], [1, 2, 3])

        # Mutating payload in dict should not mutate rec.payload
        d["payload"].append(4)
        self.assertEqual(rec.payload, [1, 2, 3])

    def test_fallback_init_fast_path(self):
        @dc._FallbackDataclass
        class FastPoint:
            x: float
            y: float
            z: float

        p1 = FastPoint(1.0, 2.0, 3.0)
        self.assertEqual((p1.x, p1.y, p1.z), (1.0, 2.0, 3.0))

        # Identity eq check
        self.assertEqual(p1, p1)


if __name__ == "__main__":
    unittest.main()
