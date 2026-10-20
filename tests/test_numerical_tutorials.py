"""小型 CPU 回归：直接抽取教程中的 NumPy 函数，不启动训练或绘图。"""
import ast
import heapq
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative, names):
    """仅执行原文件的指定定义，避免可选 PyTorch 依赖和模块级副作用。"""
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    namespace = {'np': np, 'heapq': heapq}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class NumericalTutorialTests(unittest.TestCase):
    def test_molecular_degree_matches_neighbors_and_handshake_lemma(self):
        m = load_definitions('docs/science/gnn/code/demo.py',
                             {'make_toy_molecule', 'edges_to_adj_list'})
        features, edges, labels, _, _, adj = m['make_toy_molecule']()
        np.testing.assert_array_equal(features[:, -1], [3, 3, 1, 1, 1, 1])
        self.assertEqual(features[:, -1].sum(), len(edges))
        np.testing.assert_array_equal(features[:, -1], [len(adj[i]) for i in range(6)])
        np.testing.assert_allclose(labels[:, 0], [8 / 3, 5, 6, 6, 6, 6])

    def test_huffman_internal_ids_do_not_collide_with_source_symbols(self):
        m = load_definitions('docs/information/source-coding/code/demo.py',
                             {'huffman_codes', 'avg_len', 'entropy_bits'})
        for symbols in [('A', 'B', 'C', 'D'), ('#5', 'B', 'C', 'D'), (0, 1, 2, 3)]:
            probs = dict(zip(symbols, [.4, .3, .2, .1]))
            codes = m['huffman_codes'](probs)
            self.assertEqual(set(codes), set(probs))
            for a in probs:
                for b in probs:
                    if a != b:
                        self.assertFalse(codes[b].startswith(codes[a]), codes)
            self.assertAlmostEqual(m['avg_len'](probs, codes), 1.9)
            self.assertGreaterEqual(m['avg_len'](probs, codes), m['entropy_bits'](probs))
        self.assertEqual(m['huffman_codes']({'#2': 1.0}), {'#2': '0'})

    def test_pets_planner_reward_equals_environment_reward(self):
        m = load_definitions('docs/world-models/abstract/pets/code/demo.py',
                             {'_wrap_pi', 'Pendulum', 'pendulum_reward'})
        env = m['Pendulum']()
        for theta in [0., .2, -1.1, np.pi - .01]:
            for action in [-2., -.5, 0., 1., 2.]:
                env.theta, env.omega = theta, .4
                obs, reward, _ = env.step(action)
                predicted = m['pendulum_reward'](obs[None, :], np.array([action]))[0]
                expected = (np.exp(-8 * env.theta ** 2) * np.exp(-.05 * env.omega ** 2)
                            - .01 * np.clip(action, -1., 1.) ** 2)
                self.assertAlmostEqual(reward, expected, places=13)
                self.assertAlmostEqual(predicted, reward, places=13)

    def test_pets_action_cost_affects_cem_selection(self):
        m = load_definitions('docs/world-models/abstract/pets/code/demo.py',
                             {'pendulum_reward', 'cem_plan_pendulum'})
        # 完美直立且动作不影响状态的模型：最优候选只能由真实动作成本决定。
        class FixedModel:
            n_models = 1

            def sample_next(self, states, actions, model_idx):
                return states.copy()

        np.random.seed(7)
        samples = np.clip(.5 * np.random.randn(24, 1), -1., 1.)
        expected = samples[np.argmin(np.abs(samples[:, 0])), 0]
        np.random.seed(7)
        action = m['cem_plan_pendulum'](FixedModel(), np.array([1., 0., 0.]),
                                        horizon=1, n_samples=24, n_elites=1,
                                        n_iters=1, n_particles=2)
        self.assertAlmostEqual(action, expected)


if __name__ == '__main__':
    unittest.main()
