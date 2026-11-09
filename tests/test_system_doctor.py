"""
tests/test_system_doctor.py
Unit tests verifying SystemDoctor hardware introspection, CPU fallback, and diagnostics.
"""

import json
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

from scripts.system_doctor import (
    SystemDoctor,
    probe_cuda_hardware,
    inspect_host_system,
    discover_package_managers,
    inspect_virtual_environments,
    inspect_model_weights,
)


class TestSystemDoctor(unittest.TestCase):
    def test_inspect_host_system(self):
        info = inspect_host_system()
        self.assertIn("os_name", info)
        self.assertIn("cpu_logical_cores", info)
        self.assertGreaterEqual(info["cpu_logical_cores"], 1)

    def test_probe_cuda_hardware_real(self):
        cuda_info = probe_cuda_hardware()
        self.assertIn("cuda_authorized", cuda_info)
        self.assertIn("recommended_mode", cuda_info)
        self.assertIn(cuda_info["recommended_mode"], ["CUDA_ACCELERATED", "CPU_FALLBACK"])

    @patch("shutil.which", return_value=None)
    def test_probe_cuda_hardware_fallback(self, mock_which):
        # Mock subprocess returning no CUDA
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(
                returncode=0, stdout=json.dumps({"cuda_available": False, "device_count": 0})
            )
            info = probe_cuda_hardware()
            self.assertFalse(info["cuda_authorized"])
            self.assertEqual(info["recommended_mode"], "CPU_FALLBACK")

    def test_discover_package_managers(self):
        mgrs = discover_package_managers()
        self.assertIn("micromamba", mgrs)
        self.assertIn("conda", mgrs)
        self.assertIn("mamba", mgrs)
        for name, data in mgrs.items():
            self.assertIn("installed", data)
            self.assertIn("path", data)

    def test_inspect_virtual_environments(self):
        envs = inspect_virtual_environments()
        self.assertIn("SE3nv (RFdiffusion)", envs)
        self.assertIn("mlfold (ProteinMPNN)", envs)
        self.assertIn("colabfold (AlphaFold2)", envs)

    def test_inspect_model_weights(self):
        weights = inspect_model_weights()
        self.assertIn("RFdiffusion Weights", weights)
        self.assertIn("ProteinMPNN Weights", weights)

    def test_system_doctor_card_and_dict(self):
        doc = SystemDoctor()
        d = doc.to_dict()
        self.assertIn("autoscaffold_version", d)
        self.assertEqual(d["autoscaffold_version"], "2.2.0")
        self.assertIn("verdict", d)
        self.assertIn("remediation_steps", d)

    def test_system_doctor_cli_json(self):
        py_exec = sys.executable or "python"
        res = subprocess.run(
            [py_exec, "scripts/system_doctor.py", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=10.0,
        )
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["autoscaffold_version"], "2.2.0")
        self.assertIn("host_system", data)


if __name__ == "__main__":
    unittest.main()
