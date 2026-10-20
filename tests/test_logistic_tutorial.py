"""逻辑回归教程的数值与节选回归检查。

运行：python -m unittest discover -s tests -p 'test_logistic_tutorial.py' -v
依赖：仓库现有 numpy、matplotlib、scikit-learn；无需下载数据或模型。
"""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import re
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

CHAPTER = Path(__file__).resolve().parents[1] / "docs/ml/foundations/logistic-regression"
spec = importlib.util.spec_from_file_location("logistic_tutorial_demo", CHAPTER / "code/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class LogisticTutorialTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1., -2.], [.5, 1.], [-1., .25], [2., 1.]])
        self.binary_y = np.array([0, 1, 0, 1])
        self.multi_y = np.array([0, 1, 2, 1])
        self.binary = demo.LogisticRegression()
        self.binary.w = np.array([.2, -.3])
        self.binary.b = .1
        self.multi = demo.SoftmaxRegression()
        self.multi.W = np.array([[.2, -.3, .4], [-.1, .5, .2]])
        self.multi.b = np.array([.1, -.2, .3])
        self.multi.n_classes = 3

    def test_sigmoid_extreme_finite_inputs(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            actual = demo.sigmoid(np.array([-1000., -2., 0., 2., 1000.]))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual + actual[::-1], 1., atol=1e-15)
        self.assertEqual(float(demo.sigmoid(0.)), .5)

    def test_binary_loss_not_capped_by_probability_clipping(self):
        self.binary.w = np.array([1.])
        self.binary.b = 0.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss = self.binary._compute_loss(np.array([[1000.], [-1000.]]), np.array([0, 1]))
        self.assertEqual(loss, 1000.)
        self.assertGreater(loss, -np.log(1e-15))

    def test_correct_extreme_predictions_have_finite_loss(self):
        self.binary.w = np.array([1.])
        self.binary.b = 0.
        loss = self.binary._compute_loss(np.array([[1000.], [-1000.]]), np.array([1, 0]))
        self.assertEqual(loss, 0.)

    def test_softmax_extreme_loss_and_shift_invariance(self):
        self.multi.W = np.array([[1000., -1000.]])
        self.multi.b = np.zeros(2)
        self.multi.n_classes = 2
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss = self.multi._compute_loss(np.ones((1, 1)), np.array([1]))
        self.assertEqual(loss, 2000.)
        self.multi.b += 10000.
        self.assertEqual(self.multi._compute_loss(np.ones((1, 1)), np.array([1])), loss)

    def assert_finite_difference(self, model, labels, weight_name):
        weights = getattr(model, weight_name)
        analytic_weights, analytic_bias = model._compute_gradients(self.X, labels)
        eps = 1e-6
        numerical = np.empty_like(weights)
        for idx in np.ndindex(weights.shape):
            original = weights[idx]
            weights[idx] = original + eps
            plus = model._compute_loss(self.X, labels)
            weights[idx] = original - eps
            minus = model._compute_loss(self.X, labels)
            weights[idx] = original
            numerical[idx] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(analytic_weights, numerical, atol=2e-9, rtol=2e-7)
        original_bias = np.array(model.b, copy=True)
        numerical_bias = np.empty_like(original_bias)
        for idx in np.ndindex(original_bias.shape):
            changed = original_bias.copy()
            changed[idx] += eps
            model.b = changed
            plus = model._compute_loss(self.X, labels)
            changed[idx] -= 2 * eps
            minus = model._compute_loss(self.X, labels)
            numerical_bias[idx] = (plus - minus) / (2 * eps)
        model.b = float(original_bias) if original_bias.ndim == 0 else original_bias
        np.testing.assert_allclose(analytic_bias, numerical_bias, atol=2e-9, rtol=2e-7)

    def test_binary_gradient_matches_mean_loss(self):
        self.assert_finite_difference(self.binary, self.binary_y, "w")

    def test_softmax_gradient_matches_mean_loss(self):
        self.assert_finite_difference(self.multi, self.multi_y, "W")

    def test_walkthrough_loss_excerpts_match_source(self):
        text = (CHAPTER / "code-demo.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, re.S)
        losses = [block for block in blocks if block.startswith("def _compute_loss(")]
        self.assertEqual(len(losses), 2)
        for block, model, labels in zip(losses, [self.binary, self.multi], [self.binary_y, self.multi_y]):
            namespace = {"np": np}
            exec(compile(block, "code-demo.md loss excerpt", "exec"), namespace)
            self.assertEqual(namespace["_compute_loss"](model, self.X, labels),
                             model._compute_loss(self.X, labels))

    def test_walkthrough_sigmoid_and_fit_excerpts_execute(self):
        text = (CHAPTER / "code-demo.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, re.S)
        namespace = {"np": np}
        sigmoid_block = next(block for block in blocks if block.startswith("def sigmoid("))
        exec(compile(sigmoid_block, "code-demo.md sigmoid excerpt", "exec"), namespace)
        np.testing.assert_array_equal(namespace["sigmoid"](np.array([-1000., 0., 1000.])),
                                      demo.sigmoid(np.array([-1000., 0., 1000.])))
        fit_block = next(block for block in blocks if block.startswith("def fit("))
        exec(compile(fit_block, "code-demo.md fit excerpt", "exec"), namespace)
        model = demo.LogisticRegression(max_epochs=2)
        namespace["fit"](model, self.X, self.binary_y, verbose=False)
        self.assertEqual(model.w.shape, (2,))
        self.assertTrue(np.isfinite(model.w).all())

    def test_iris_demo_step_size_decreases_loss(self):
        X, y = demo.load_iris(return_X_y=True)
        tree = ast.parse((CHAPTER / "code/demo.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        configurations = {}
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("LogisticRegression", "SoftmaxRegression"):
                    configurations[node.func.id] = {
                        keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
        for binary in (True, False):
            mask = y < 2 if binary else np.ones(len(y), dtype=bool)
            X_train, X_test, y_train, y_test = demo.train_test_split(
                X[mask, :2], y[mask], test_size=.2, random_state=42)
            np.random.seed(42)
            cls = demo.LogisticRegression if binary else demo.SoftmaxRegression
            model = cls(**configurations[cls.__name__])
            model.fit(X_train, y_train, verbose=False)
            final_loss = model._compute_loss(X_train, y_train)
            losses = np.r_[model.loss_history, final_loss]
            self.assertTrue(np.isfinite(losses).all())
            self.assertTrue(np.all(np.diff(losses) <= 1e-12))
            self.assertGreaterEqual(np.mean(model.predict(X_test) == y_test), .85)
            augmented = np.c_[X_train, np.ones(len(X_train))]
            bound = np.linalg.norm(augmented, 2) ** 2 / len(X_train)
            bound *= .25 if binary else .5
            self.assertLessEqual(model.learning_rate, 1 / bound)

    def test_reported_final_loss_uses_final_parameters(self):
        for cls, labels in [(demo.LogisticRegression, self.binary_y),
                            (demo.SoftmaxRegression, self.multi_y)]:
            np.random.seed(42)
            model = cls(learning_rate=.5, max_epochs=1)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                model.fit(self.X, labels)
            final_loss = model._compute_loss(self.X, labels)
            self.assertIn(f"最终损失: {final_loss:.6f}", out.getvalue())
            self.assertNotAlmostEqual(final_loss, model.loss_history[-1], places=6)


if __name__ == "__main__":
    unittest.main()
