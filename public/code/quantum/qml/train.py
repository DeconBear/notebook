"""
train.py：训练入口脚本。

迁自 https://github.com/DeconBear/qml-mnist-classify （MIT）。
需要 pyvqnet。请在本目录运行：python train.py

这个目录只保留一个固定配置。

运行方式：
    python train.py

可选：
    python train.py --artifact-dir alt_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qml_core import default_artifact_dir, get_default_spec, train_experiment


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    当前目录只保留一个可选参数：
    - artifact_dir：手动指定模型和报告输出目录，建议传相对路径
    """

    parser = argparse.ArgumentParser(description="训练当前配置。")
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="可选：手动指定模型和训练报告输出目录，建议使用相对路径。留空则保存在 model 目录。",
    )
    return parser.parse_args()


def main() -> None:
    """
    训练入口。
    """

    args = parse_args()
    # root：
    # 当前运行目录。按约定应当在 code 目录内执行脚本，
    # 这样 dataset/ 和 model/ 都能通过相对路径直接访问。
    root = Path(".")
    spec = get_default_spec()
    output_dir = Path(args.artifact_dir) if args.artifact_dir else default_artifact_dir(root)

    print(f"Start training: {spec.model.display_name}")
    print(f"Output directory: {output_dir}")

    report = train_experiment(spec, root=root, output_dir=output_dir, verbose=True)

    print("Training finished.")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Train size: {report['train_size']}")
    print(f"Validation size: {report['validation_size']}")
    print(f"Best validation accuracy: {report['best_validation_accuracy']:.4f}")
    print(f"Encoding gate count: {report['encoding_gate_count']}")
    print(f"Encoding depth: {report['encoding_depth']}")
    print(f"Estimated complexity score: {report['encoding_complexity_score']:.4f}")
    print(
        "Estimated validation score: "
        f"{report['estimated_validation_total_score']:.4f}"
    )
    print(f"Model saved to: {report['model_path']}")
    print(f"Report saved to: {report['report_path']}")


if __name__ == "__main__":
    main()
