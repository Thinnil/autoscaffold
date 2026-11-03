"""
tests/test_versioning.py
Unit tests verifying AutoScaffold unified semantic versioning (v2.2.0).
"""

import subprocess
import sys
import unittest
from scripts.version import __version__, __version_info__, get_version, get_git_sync_status
from scripts.methodology_reporter import AUTOSCAFFOLD_VERSION
import scripts.chimera_scaffold as cs
import scripts.run_chimera_pipeline as rp


class TestVersioning(unittest.TestCase):
    def test_version_constants(self):
        self.assertEqual(__version__, "2.2.0")
        self.assertEqual(__version_info__, (2, 2, 0))
        self.assertEqual(get_version(), "2.2.0")

    def test_module_version_synchronization(self):
        self.assertEqual(AUTOSCAFFOLD_VERSION, "2.2.0")
        self.assertEqual(cs.__version__, "2.2.0")
        self.assertEqual(rp.__version__, "2.2.0")

    def test_git_sync_status(self):
        status = get_git_sync_status()
        self.assertIsInstance(status, str)
        self.assertTrue(len(status) > 0)

    def test_cli_version_flags(self):
        tools = [
            "scripts/run_chimera_pipeline.py",
            "scripts/chimera_scaffold.py",
            "scripts/system_doctor.py",
        ]
        py_exec = sys.executable or "python"
        for tool in tools:
            res = subprocess.run(
                [py_exec, tool, "--version"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            self.assertEqual(res.returncode, 0, f"Tool {tool} --version failed with exit code {res.returncode}")
            output = (res.stdout + res.stderr).strip()
            self.assertIn("2.2.0", output, f"Tool {tool} --version did not report 2.2.0: {output}")


if __name__ == "__main__":
    unittest.main()
