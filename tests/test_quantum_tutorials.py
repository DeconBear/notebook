# -*- coding: utf-8 -*-
"""量子教程的小型 CPU 回归：解析解、测量分支与误差阶，无训练或下载。"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'quantum'


def load_demo(chapter):
    spec = importlib.util.spec_from_file_location(f'quantum_{chapter}', ROOT / chapter / 'code' / 'demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantumTutorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_demo('network')
        cls.memory = load_demo('memory')
        cls.simulation = load_demo('simulation')
        cls.overview = load_demo('overview')

    def test_teleport_all_branches_and_complex_inputs(self):
        rng = np.random.default_rng(42)
        states = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1j]) / np.sqrt(2)]
        for _ in range(20):
            psi = rng.normal(size=2) + 1j * rng.normal(size=2)
            states.append(psi / np.linalg.norm(psi))
        for psi in states:
            for branch in range(4):
                with patch.object(self.network.np.random, 'choice', return_value=branch) as choose:
                    bob = self.network.teleport_once(psi)
                self.assertEqual(choose.call_args.args[0], 4)
                np.testing.assert_allclose(choose.call_args.kwargs['p'], np.full(4, 0.25), atol=1e-14)
                self.assertAlmostEqual(self.network.fidelity(psi, bob), 1.0, places=13)

    def test_teleport_rejects_invalid_state(self):
        for psi in ([1, 1], [0, 0], [1, 0, 0], [float('nan'), 0]):
            with self.assertRaises(ValueError):
                self.network.teleport_once(psi)

    def test_bob_without_classical_message_is_maximally_mixed(self):
        psi = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
        rho = np.outer(psi, psi.conj())
        n = self.network
        uncorrected = sum(u @ rho @ u.conj().T for u in [n.I2, n.X, n.Z, n.X @ n.Z]) / 4
        np.testing.assert_allclose(uncorrected, np.eye(2) / 2, atol=1e-14)

    def test_memory_matches_analytic_decay_and_fidelity(self):
        m = self.memory
        plus = np.array([1, 1]) / np.sqrt(2)
        rho = np.outer(plus, plus)
        for t in [0, 0.01, 0.4, 1, 2.5, 20]:
            out = m.dephase(m.amp_damping(rho, t), t)
            self.assertAlmostEqual(out[0, 1].real, 0.5 * np.exp(-t / m.T2), places=14)
            self.assertAlmostEqual(m.fidelity_pure(out, plus), (1 + np.exp(-t / m.T2)) / 2, places=14)
            self.assertAlmostEqual(np.trace(out).real, 1, places=14)
            self.assertGreaterEqual(np.linalg.eigvalsh(out).min(), -1e-14)
            excited = m.amp_damping(np.diag([0, 1]), t)
            self.assertAlmostEqual(excited[1, 1].real, np.exp(-t / m.T1), places=14)

    def test_memory_no_noise_limit_and_input_unchanged(self):
        m = self.memory
        rho = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
        before = rho.copy()
        np.testing.assert_allclose(m.dephase(m.amp_damping(rho, 2, np.inf), 2, np.inf), rho)
        np.testing.assert_array_equal(rho, before)
        # 整数矩阵输入也必须在浮点/复数域衰减，不能截断或发生 dtype 错误。
        out = m.dephase(np.array([[1, 1], [1, 1]]), 1, 1)
        self.assertAlmostEqual(out[0, 1].real, np.exp(-1))
        for time, tau in [(-1, 1), (1, 0), (1, -1), (np.inf, 1), (1, np.nan)]:
            for channel in (m.amp_damping, m.dephase):
                with self.assertRaises(ValueError):
                    channel(rho, time, tau)

    def test_bb84_sampling_and_empty_sift(self):
        np.random.seed(42)
        self.assertEqual(self.network.bb84_qber(100_000), 0.0)
        self.assertLess(abs(self.network.bb84_qber(100_000, eve=True) - 0.25), 0.01)
        with patch.object(self.network.np.random, 'randint', side_effect=[np.array([0]), np.array([0]), np.array([1])]):
            self.assertTrue(np.isnan(self.network.bb84_qber(1)))
        for n in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                self.network.bb84_qber(n)

    def test_trotter_unitarity_and_infidelity_order(self):
        s = self.simulation
        psi = np.array([1, 0, 0, 0], dtype=complex)
        exact = s.expm_herm(s.Htot, s.T) @ psi
        errors = []
        for steps in [32, 64, 128]:
            u = s.trotter(steps)
            np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)
            errors.append(1 - abs(np.vdot(exact, u @ psi)) ** 2)
        np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=0.05)
        for steps in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                s.trotter(steps)

    def test_superposition_vs_mixture(self):
        h = self.overview.H
        plus = np.ones(2) / np.sqrt(2)
        pure = np.outer(plus, plus)
        mix = np.eye(2) / 2
        np.testing.assert_allclose(np.diag(pure), np.diag(mix))
        np.testing.assert_allclose(h @ pure @ h.conj().T, np.diag([1, 0]), atol=1e-14)
        np.testing.assert_allclose(h @ mix @ h.conj().T, mix, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
