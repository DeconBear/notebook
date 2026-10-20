"""教程源码语法检查；不执行可能下载权重或训练的脚本。"""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TutorialSyntaxTests(unittest.TestCase):
    def test_all_python_sources_parse(self):
        files = sorted((ROOT / 'docs').rglob('*.py'))
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

    def test_optimizer_exercise_reports_unfinished_tasks(self):
        path = ROOT / 'docs/nn-decision/dl/optimizers/code/exercise.py'
        result = subprocess.run([sys.executable, str(path)], capture_output=True,
                                text=True, timeout=30, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('TODO 未完成'), 3)
        self.assertIn('所有测试完成', result.stdout)


if __name__ == '__main__':
    unittest.main()
