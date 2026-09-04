"""
eval.py：评估入口脚本。

迁自独立示例工程 qml-mnist-classify（MIT），现随本章发布。
需要 pyvqnet。请在本目录运行：python eval.py

这个目录只保留一个固定配置。

运行方式：
    python eval.py

可选：
    python eval.py --artifact-dir alt_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qml_core import default_artifact_dir, evaluate_saved_model, get_default_spec


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    当前目录只保留一个可选参数：
    - artifact_dir：手动指定模型与报告目录，建议传相对路径
    """

    parser = argparse.ArgumentParser(description="评估当前配置。")
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="可选：手动指定模型与训练报告所在目录，建议使用相对路径。留空则使用 model 目录。",
    )
    return parser.parse_args()


def main() -> None:
    """
    评估入口。
    """

    args = parse_args()
    # root：
    # 当前运行目录。按约定应当在 code 目录内执行脚本，
    # 这样 dataset/ 和 model/ 都能通过相对路径直接访问。
    root = Path(".")
    spec = get_default_spec()
    output_dir = Path(args.artifact_dir) if args.artifact_dir else default_artifact_dir(root)

    print(f"Start evaluating: {spec.model.display_name}")
    print(f"Artifact directory: {output_dir}")

    result = evaluate_saved_model(root=root, output_dir=output_dir, verbose=True)

    print("Evaluation finished.")
    print(f"Scheme name: {result['scheme_name']}")
    print(f"Model path: {result['model_path']}")
    print(f"Report path: {result['report_path']}")
    print(f"Prediction table path: {result['prediction_table_path']}")


if __name__ == "__main__":
    main()
