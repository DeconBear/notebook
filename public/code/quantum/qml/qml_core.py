"""
qml_core.py：本项目的“共享核心模块”。

迁自独立示例工程 qml-mnist-classify（MIT License），现随本章发布。
未安装 pyvqnet 时请不要直接 import 本文件；见同目录 demo.py 的降级逻辑。

如果你把整个项目想成一台机器，那么：

- `train.py` 像“启动按钮”
- `eval.py` 像“验收按钮”
- `run_experiments.py` 像“批量实验控制台”
- `qml_core.py` 就像“发动机和传动系统”

也就是说，真正和模型、训练、评估、方案配置强相关的核心逻辑，
都集中在这个文件里。

为什么要这样设计？

因为如果我们把核心逻辑分别写在 train.py / eval.py / run_experiments.py 中，
会出现几个明显问题：

1. 训练脚本和评估脚本各写一份模型结构，容易写着写着不一致
2. 想增加新方案时，要改很多地方，容易漏掉
3. 做对比实验时，会出现“不同脚本隐式逻辑不同”的风险
4. 初学者阅读时，也不容易找到“真正关键的代码在哪里”

因此，这个文件的目标是：

1. 统一保存常量
2. 统一描述方案配置
3. 统一定义模型结构
4. 统一实现训练逻辑
5. 统一实现评估逻辑
6. 统一输出实验结果格式


"""

from __future__ import annotations

# json：
# 用来读写训练报告 JSON 文件。
# 例如训练完成后，我们会把最佳准确率、门数、深度、配置等信息写进 JSON。
import csv
import json

# random：
# Python 自带随机数模块。
# 训练时为了尽量复现结果，需要固定随机种子。
import random
from datetime import datetime

# dataclass：
# 用来定义“只负责装配置数据”的轻量类。
# 比如 ModelConfig、TrainingConfig、ExperimentSpec 都很适合用 dataclass。
#
# asdict：
# 用来把 dataclass 对象转换成普通字典，便于写 JSON。
from dataclasses import asdict, dataclass

# Path：
# 用来处理文件路径，比如：
# - 数据集路径
# - 模型保存路径
# - 报告保存路径
from pathlib import Path

# numpy：
# 用来做数值处理，比如：
# - 图像数组预处理
# - 精度统计
# - argmax 取预测类别
import numpy as np

# pyvqnet.nn：
# VQNet 的神经网络模块，里面有：
# - Linear
# - Tanh
# - Sigmoid
# - CrossEntropyLoss
import pyvqnet.nn as nn

# pyvqnet.optim：
# VQNet 的优化器模块，这里我们主要用 AdamW。
import pyvqnet.optim as optim

# QTensor：
# VQNet 使用的张量类型。
# 当数据要进入模型、参与自动求导时，通常要转成 QTensor。
#
# tensor：
# VQNet 的张量运算工具模块，这里主要用 reshape、mean 等操作。
from pyvqnet import QTensor, tensor

# data_generator：
# VQNet 提供的数据迭代器，用来把 numpy 数据分成 batch。
from pyvqnet.data import data_generator

# 量子门和测量相关组件：
# - RY / RZ：单比特旋转门
# - CNOT：双比特纠缠门
# - MeasureAll：测量所有量子比特
# - QMachine：量子模拟器/量子设备对象
from pyvqnet.qnn.vqc import CNOT, MeasureAll, QMachine, RY, RZ

# VQC_HardwareEfficientAnsatz：
# 固定 Hardware Efficient Ansatz。
from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz

# save_parameters / load_parameters：
# 用于保存和加载模型参数。
from pyvqnet.utils.storage import load_parameters, save_parameters


# ============================================================
# 一、全局常量：这些值是整个项目都会频繁用到的“固定信息”
# ============================================================

# NUM_QUBITS：
# 量子比特数量。
# 当前实现固定使用 8 比特量子线路，所以这里写死为 8。
NUM_QUBITS = 8

# IMG_SIZE：
# 图像边长。
# 当前使用的图像已经被缩放为 16x16。
IMG_SIZE = 16

# IMG_PIXELS：
# 图像总像素数。
# 因为 16x16 = 256，所以总维度是 256。
# 后续做经典压缩时，我们会先把图像展平到这个维度。
IMG_PIXELS = IMG_SIZE * IMG_SIZE

# TWO_PI：
# 2π，对应一整圈弧度。
# 量子旋转门的角度一般使用弧度制，因此常常需要把 [0,1] 的输出缩放到 [0, 2π]。
TWO_PI = 2.0 * np.pi

# DEFAULT_MODEL_NAME：
# 当前默认配置的内部名字。
DEFAULT_MODEL_NAME = "lean_reupload"


@dataclass(frozen=True)
class ModelConfig:
    """
    ModelConfig：描述“模型结构长什么样”的配置对象。

    这个类只负责存放“结构设计相关”的信息，
    不负责训练，不负责前向传播，也不负责保存模型。

    你可以把它理解成“建筑蓝图”：
    它描述的是一套模型要怎么搭，而不是模型已经训练好了什么参数。
    """

    # scheme_name：方案内部名字，主要给代码和文件夹使用。
    scheme_name: str

    # display_name：展示给人看的名字，主要用于命令行、报告和教程。
    display_name: str

    # description：方案的自然语言说明。
    description: str

    # encoder_dims：经典压缩器的隐藏层结构。
    # 例如：
    # - (64,) 表示只用一层 256 -> 64
    # - (128, 64) 表示两层 256 -> 128 -> 64
    encoder_dims: tuple[int, ...]

    # use_two_axis_encoding：
    # True 表示用 Ry + Rz 双轴编码。
    # False 表示只用 Ry 单轴编码。
    use_two_axis_encoding: bool

    # use_encoding_entangle_chain：
    # True 表示编码阶段还要额外加一条线性 CNOT 链。
    # False 表示编码阶段不额外加 CNOT 链。
    use_encoding_entangle_chain: bool

    # ansatz_depth：固定 HEA 的深度。
    ansatz_depth: int = 2

    @property
    def encoding_gate_count(self) -> int:
        """
        计算当前方案“编码部分”的门数 G。

        注意：
        这里只统计自定义编码线路，
        不统计固定 HEA 后端。
        """

        # gate_count：当前累计门数。
        # 先把所有方案必然拥有的 8 个 Ry 算进去。
        gate_count = NUM_QUBITS

        # 如果编码阶段启用了 CNOT 链，就再加 7 个门。
        if self.use_encoding_entangle_chain:
            gate_count += NUM_QUBITS - 1

        # 如果是双轴编码，再加 8 个 Rz。
        if self.use_two_axis_encoding:
            gate_count += NUM_QUBITS

        return gate_count

    @property
    def encoding_depth(self) -> int:
        """
        估算当前方案“编码部分”的线路深度 D。

        为了保持报告和代码口径一致，这里采用一个简洁、可解释的估算方式：

        - 第一层 Ry 算 1 层
        - 如果有线性 CNOT 链，则近似算 2 层
        - 如果有 Rz，再算 1 层
        """

        # depth：当前累计深度。
        # 因为所有方案都有一层 Ry，所以从 1 开始。
        depth = 1

        if self.use_encoding_entangle_chain:
            depth += 2

        if self.use_two_axis_encoding:
            depth += 1

        return depth


@dataclass(frozen=True)
class TrainingConfig:
    """
    TrainingConfig：描述“训练超参数”的配置对象。

    这个类只负责装训练相关数字，不负责模型结构。
    """

    # epochs：训练轮数。
    epochs: int

    # batch_size：每次喂给模型的样本数。
    batch_size: int

    # learning_rate：学习率。
    learning_rate: float

    # weight_decay：权重衰减。
    weight_decay: float

    # alignment_weight：对齐正则项的权重。
    alignment_weight: float

    # seed：随机种子。
    seed: int

    # validation_ratio：
    # 从训练集中切出多少比例作为内部验证集。
    validation_ratio: float


@dataclass(frozen=True)
class ExperimentSpec:
    """
    ExperimentSpec：把“模型结构配置”和“训练配置”打包成一个完整实验。

    为什么需要这个类？
    因为一个实验不是只有模型结构，也不是只有训练超参数，
    它应该把“这次实验完整要怎么跑”统一描述清楚。
    """

    # experiment_name：实验唯一名字，主要用于实验目录和对比表。
    experiment_name: str

    # model：本次实验采用的模型结构配置。
    model: ModelConfig

    # training：本次实验采用的训练超参数配置。
    training: TrainingConfig

    # notes：对这个实验的补充说明。
    notes: str = ""


def set_seed(seed: int) -> None:
    """
    固定随机种子，尽量提高复现性。
    """

    random.seed(seed)
    np.random.seed(seed)


def estimate_complexity_score(gate_count: int, depth: int) -> float:
    """
    按当前使用的复杂度公式计算编码得分。
    """

    if gate_count <= 0 or depth <= 0:
        return 0.0

    return 20.0 * (1000.0 / (gate_count + 1000.0) + 500.0 / (depth + 500.0))


def estimate_total_score(accuracy: float, gate_count: int, depth: int) -> float:
    """
    估算客观部分总分。

    这里只统计：
    - 准确率部分：30 * accuracy
    - 复杂度部分：按当前公式计算
    """

    return 30.0 * accuracy + estimate_complexity_score(gate_count, depth)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    对输入图像做预处理。

    这里会做两件事：
    1. 转成 float32
    2. 把像素值从 [0,1] 映射到 [-1,1]
    """

    images = images.astype(np.float32)
    return (images - 0.5) / 0.5


def load_train_dataset(root: Path):
    """
    读取训练集。

    参数：
    - root：代码工作目录
    """

    # dataset_dir：
    # 数据集子目录。
    dataset_dir = root / "dataset"

    # train_data：训练集 npz 文件对象。
    train_data = np.load(
        dataset_dir / f"mnist_train_1000_{IMG_SIZE}_{IMG_SIZE}.npz"
    )

    # x_train：图像数据，需要先做预处理。
    x_train = preprocess_images(train_data["data"])

    # y_train：标签数据，转成 int64 便于分类任务使用。
    y_train = train_data["label"].astype(np.int64)

    return x_train, y_train


def load_test_dataset(root: Path):
    """
    读取测试集。

    参数：
    - root：代码工作目录
    """

    dataset_dir = root / "dataset"
    test_data = np.load(
        dataset_dir / f"mnist_test_200_{IMG_SIZE}_{IMG_SIZE}.npz"
    )

    x_test = preprocess_images(test_data["data"])
    y_test = test_data["label"].astype(np.int64)

    return x_test, y_test


def split_train_validation(
    x_data: np.ndarray,
    y_data: np.ndarray,
    validation_ratio: float,
    seed: int,
):
    """
    从训练集中按类别分层切出内部验证集。

    参数：
    - x_data：完整训练图像
    - y_data：完整训练标签
    - validation_ratio：验证集比例
    - seed：随机种子
    """

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    rng = np.random.RandomState(seed)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []

    for label in np.unique(y_data):
        label_indices = np.where(y_data == label)[0].copy()
        rng.shuffle(label_indices)

        validation_count = max(1, int(round(label_indices.shape[0] * validation_ratio)))
        validation_indices.append(label_indices[:validation_count])
        train_indices.append(label_indices[validation_count:])

    train_index_array = np.concatenate(train_indices)
    validation_index_array = np.concatenate(validation_indices)

    rng.shuffle(train_index_array)
    rng.shuffle(validation_index_array)

    return (
        x_data[train_index_array],
        y_data[train_index_array],
        x_data[validation_index_array],
        y_data[validation_index_array],
    )


def ensure_dir(path: Path) -> None:
    """
    确保某个目录存在。
    """

    path.mkdir(parents=True, exist_ok=True)


def default_artifact_dir(root: Path) -> Path:
    """
    默认把训练产物保存在 model 子目录。
    """
    return root / "model"


def default_output_dir(root: Path) -> Path:
    """
    默认把评估摘要保存在 output 子目录。
    """
    return root / "output"


def save_json(path: Path, payload: dict) -> None:
    """
    统一用 UTF-8 保存 JSON。
    """

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_prediction_rows(path: Path, rows: list[dict]) -> Path:
    """
    把逐样本预测结果保存成 CSV 表格。

    参数：
    - path：CSV 文件路径
    - rows：每一行的预测信息
    """

    if not rows:
        headers = [
            "sample_index",
            "true_label",
            "true_digit",
            "predicted_label",
            "predicted_digit",
            "is_correct",
            "logit_0",
            "logit_1",
        ]
    else:
        headers = list(rows[0].keys())

    def _write_csv(target_path: Path) -> None:
        with target_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    try:
        _write_csv(path)
        return path
    except PermissionError:
        # 如果默认文件正被其他程序占用，就退回到一个带时间戳的新文件，
        # 避免整个评估流程因为文件锁而中断。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        _write_csv(fallback_path)
        return fallback_path


def label_to_digit(label: int) -> int:
    """
    把二分类标签映射回原始手写数字。

    当前数据集中：
    - 标签 0 对应数字 3
    - 标签 1 对应数字 6
    """

    return 3 if int(label) == 0 else 6

def get_default_spec() -> ExperimentSpec:
    """
    返回当前目录唯一保留的固定配置。
    """
    return ExperimentSpec(
        experiment_name="lean_reupload",
        model=ModelConfig(
            scheme_name="lean_reupload",
            display_name="精简双轴重上传编码",
            description=(
                "保留 Ry + Rz 双轴编码，但移除编码阶段的 CNOT 链，"
                "让纠缠主要由固定 HEA 承担，以进一步降低编码复杂度。"
            ),
            encoder_dims=(128, 64),
            use_two_axis_encoding=True,
            use_encoding_entangle_chain=False,
            ansatz_depth=2,
        ),
        training=TrainingConfig(
            epochs=24,
            batch_size=128,
            learning_rate=0.005,
            weight_decay=1e-4,
            alignment_weight=0.01,
            seed=42,
            validation_ratio=0.2,
        ),
        notes="训练阶段仅使用训练集及其内部验证集，测试集只在独立评估时使用。",
    )


class QuantumImageClassifier(nn.Module):
    """
    QuantumImageClassifier：可切换编码方案的混合量子-经典分类器。

    这是整个项目里最核心的模型类。

    你可以把它拆成 5 个部分来理解：

    1. 经典压缩器
    2. 角度头
    3. 自定义量子编码
    4. 固定 HEA
    5. 测量 + 线性分类头
    """

    def __init__(self, model_config: ModelConfig, name: str = ""):
        """
        初始化模型。

        参数：
        - model_config：这套模型要采用的结构配置
        - name：模块名字，通常保持默认空字符串即可
        """

        super().__init__(name)

        # self.model_config：
        # 把用户传入的结构配置保存下来。
        # 后面 forward、_encode 都要反复用到这些配置信息。
        self.model_config = model_config

        # ====================================================
        # 一、经典压缩器
        # ====================================================

        # self.encoder_layer_1：
        # 第一层经典线性压缩层。
        # 输入固定是 256 维图像，输出维度由配置决定。
        self.encoder_layer_1 = nn.Linear(IMG_PIXELS, model_config.encoder_dims[0])

        # self.encoder_activation_1：
        # 第一层压缩后的激活函数。
        self.encoder_activation_1 = nn.Tanh()

        # self.encoder_layer_2 / self.encoder_activation_2：
        # 第二层经典压缩器，默认先设为 None。
        # 如果配置里没有第二层，就保持 None。
        self.encoder_layer_2 = None
        self.encoder_activation_2 = None

        # encoder_output_dim：
        # 经典压缩器最终输出维度。
        # 后面的角度头需要知道自己接收多少维输入。
        encoder_output_dim = model_config.encoder_dims[0]

        # 如果配置中给了第二层维度，就额外创建第二层。
        if len(model_config.encoder_dims) > 1:
            self.encoder_layer_2 = nn.Linear(
                model_config.encoder_dims[0],
                model_config.encoder_dims[1],
            )
            self.encoder_activation_2 = nn.Tanh()
            encoder_output_dim = model_config.encoder_dims[1]

        # ====================================================
        # 二、角度头：把经典特征映射成量子角度
        # ====================================================

        # self.angle_head_y：
        # 所有方案都会有一组 Ry 角度，所以它一定存在。
        self.angle_head_y = nn.Linear(encoder_output_dim, NUM_QUBITS)

        # self.angle_head_z：
        # 只有双轴编码方案才需要第二组 Rz 角度。
        self.angle_head_z = None
        if model_config.use_two_axis_encoding:
            self.angle_head_z = nn.Linear(encoder_output_dim, NUM_QUBITS)

        # self.angle_activation：
        # 把角度头输出限制到 [0,1]，后面再缩放到 [0, 2π]。
        self.angle_activation = nn.Sigmoid()

        # ====================================================
        # 三、固定量子变分后端：固定 HEA
        # ====================================================

        # self.ansatz：
        # 固定 Hardware Efficient Ansatz。
        self.ansatz = VQC_HardwareEfficientAnsatz(
            NUM_QUBITS,
            ["rx", "ry", "rz"],
            entangle_gate="CNOT",
            entangle_rules="linear",
            depth=model_config.ansatz_depth,
        )

        # self.measure：
        # 测量所有量子比特在 Z 基上的期望值。
        self.measure = MeasureAll(obs=[{f"Z{idx}": 1.0} for idx in range(NUM_QUBITS)])

        # self.classifier：
        # 最终线性分类头，把 8 维量子特征映射到 2 类 logits。
        self.classifier = nn.Linear(NUM_QUBITS, 2)

        # self.device：
        # 量子模拟器对象，真正执行量子门并维护量子态。
        self.device = QMachine(NUM_QUBITS)

    def _encode(
        self,
        angles_y: QTensor,
        angles_z: QTensor | None,
        batch_size: int,
    ) -> None:
        """
        根据当前方案配置执行编码线路。

        参数：
        - angles_y：第一组角度，给 Ry 使用
        - angles_z：第二组角度，给 Rz 使用，若方案不需要则为 None
        - batch_size：当前 batch 的样本数
        """

        # 每处理一个新 batch，都要重置量子态。
        self.device.reset_states(batch_size)

        # 所有方案都先做一轮 8 个 Ry 编码。
        for qubit in range(NUM_QUBITS):
            gate = RY(
                has_params=True,
                trainable=False,
                init_params=angles_y[:, qubit : qubit + 1],
                wires=qubit,
            )
            gate(q_machine=self.device)

        # 如果该方案要求在编码阶段加入纠缠链，就执行线性 CNOT。
        if self.model_config.use_encoding_entangle_chain:
            for qubit in range(NUM_QUBITS - 1):
                CNOT(wires=[qubit, qubit + 1], q_machine=self.device)

        # 如果是双轴方案，就额外执行一轮 Rz 编码。
        if self.model_config.use_two_axis_encoding and angles_z is not None:
            for qubit in range(NUM_QUBITS):
                gate = RZ(
                    has_params=True,
                    trainable=False,
                    init_params=angles_z[:, qubit : qubit + 1],
                    wires=qubit,
                )
                gate(q_machine=self.device)

    def _encode_classical_features(self, x: QTensor) -> QTensor:
        """
        执行经典压缩器部分。

        参数：
        - x：展平后的图像张量，形状大致为 [batch_size, 256]

        返回值：
        - hidden：压缩后的隐藏特征
        """

        # hidden：先经过第一层线性压缩，再经过激活函数。
        hidden = self.encoder_activation_1(self.encoder_layer_1(x))

        # 如果当前方案定义了第二层压缩器，就继续通过第二层。
        if self.encoder_layer_2 is not None and self.encoder_activation_2 is not None:
            hidden = self.encoder_activation_2(self.encoder_layer_2(hidden))

        return hidden

    def forward(self, x: QTensor, return_aux: bool = False):
        """
        模型完整前向传播。

        参数：
        - x：输入图像张量，通常形状为 [batch_size, 16, 16]
        - return_aux：是否额外返回辅助量
        """

        # 先把图像展平为 [batch_size, 256]。
        x = tensor.reshape(x, [x.shape[0], IMG_PIXELS])

        # hidden：经典压缩后的隐藏特征。
        hidden = self._encode_classical_features(x)

        # angles_y：第一组量子编码角，给 Ry 使用。
        angles_y = TWO_PI * self.angle_activation(self.angle_head_y(hidden))

        # angles_z：第二组量子编码角，给 Rz 使用。
        angles_z = None
        if self.angle_head_z is not None:
            angles_z = TWO_PI * self.angle_activation(self.angle_head_z(hidden))

        # 执行自定义量子编码。
        self._encode(angles_y, angles_z, x.shape[0])

        # 经过固定 HEA。
        self.ansatz(q_machine=self.device)

        # quantum_features：测量得到的 8 维量子特征。
        quantum_features = self.measure(q_machine=self.device)

        # logits：分类头输出的两个类别分数。
        logits = self.classifier(quantum_features)

        # 如果不需要辅助输出，就直接返回 logits。
        if not return_aux:
            return logits

        # alignment_target：
        # 对齐正则目标，只有双轴编码时才有意义。
        alignment_target = None
        if angles_z is not None:
            alignment_target = ((angles_y + angles_z) / TWO_PI) - 1.0

        return logits, quantum_features, alignment_target


def compute_alignment_loss(
    quantum_features: QTensor,
    alignment_target: QTensor | None,
) -> QTensor | None:
    """
    计算对齐正则损失。

    参数：
    - quantum_features：量子测量得到的 8 维特征
    - alignment_target：希望量子特征对齐到的目标，若没有则为 None
    """

    if alignment_target is None:
        return None

    # diff：量子特征与目标之间的差值。
    diff = quantum_features - alignment_target

    # 用最简单的均方误差 MSE 作为对齐损失。
    return tensor.mean(diff * diff)


def evaluate_accuracy(
    model: QuantumImageClassifier,
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
) -> float:
    """
    计算模型在指定数据集上的准确率。

    参数：
    - model：要评估的模型
    - x_data：图像数据
    - y_data：标签数据
    - batch_size：评估时每次送多少样本
    """

    # 切换到评估模式。
    model.eval()

    # correct：累计预测正确的样本数。
    correct = 0

    # total：累计评估过的样本数。
    total = 0

    for x_batch, y_batch in data_generator(
        x_data,
        y_data,
        batch_size=batch_size,
        shuffle=False,
    ):
        # logits：当前 batch 的模型输出。
        logits = model(QTensor(x_batch))

        # predictions：取两个类别分数中最大的那个索引作为预测类别。
        predictions = np.argmax(logits.numpy(), axis=1)

        correct += int(np.sum(predictions == y_batch))
        total += int(y_batch.shape[0])

    return correct / max(total, 1)


def collect_prediction_rows(
    model: QuantumImageClassifier,
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
) -> tuple[float, list[dict]]:
    """
    收集逐样本预测结果，并返回准确率与表格行。

    参数：
    - model：要评估的模型
    - x_data：图像数据
    - y_data：标签数据
    - batch_size：评估时每次送多少样本
    """

    model.eval()

    rows: list[dict] = []
    correct = 0
    total = 0
    sample_index = 0

    for x_batch, y_batch in data_generator(
        x_data,
        y_data,
        batch_size=batch_size,
        shuffle=False,
    ):
        logits = model(QTensor(x_batch))
        logits_np = logits.numpy()
        predictions = np.argmax(logits_np, axis=1)

        batch_size_now = int(y_batch.shape[0])

        for local_index in range(batch_size_now):
            true_label = int(y_batch[local_index])
            predicted_label = int(predictions[local_index])
            is_correct = int(predicted_label == true_label)

            rows.append(
                {
                    "sample_index": sample_index,
                    "true_label": true_label,
                    "true_digit": label_to_digit(true_label),
                    "predicted_label": predicted_label,
                    "predicted_digit": label_to_digit(predicted_label),
                    "is_correct": is_correct,
                    "logit_0": float(logits_np[local_index, 0]),
                    "logit_1": float(logits_np[local_index, 1]),
                }
            )

            correct += is_correct
            total += 1
            sample_index += 1

    accuracy = correct / max(total, 1)
    return accuracy, rows


def train_experiment(
    spec: ExperimentSpec,
    root: Path,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    训练一个完整实验方案，并返回结构化结果。

    参数：
    - spec：本次实验的完整配置
    - root：数据目录 / 工作目录
    - output_dir：输出目录，如果为 None 则按默认规则自动决定
    - verbose：是否打印详细训练日志
    """

    # 先固定随机种子，尽量让结果可复现。
    set_seed(spec.training.seed)

    # 读取完整数据。
    x_train_all, y_train_all = load_train_dataset(root)

    # 从训练集中再切出一份内部验证集，用于选择最佳模型。
    x_train, y_train, x_validation, y_validation = split_train_validation(
        x_train_all,
        y_train_all,
        validation_ratio=spec.training.validation_ratio,
        seed=spec.training.seed,
    )

    # 如果用户没手动指定输出目录，就用默认规则自动决定。
    if output_dir is None:
        output_dir = default_artifact_dir(root)

    # 确保输出目录存在。
    ensure_dir(output_dir)

    # model_path：最优模型参数保存路径。
    model_path = output_dir / "best_model.vq"

    # report_path：训练报告 JSON 保存路径。
    report_path = output_dir / "training_report.json"

    # model：真正要训练的模型实例。
    model = QuantumImageClassifier(spec.model)

    # optimizer：优化器，负责根据梯度更新参数。
    optimizer = optim.AdamW(
        model.parameters(),
        lr=spec.training.learning_rate,
        weight_decay=spec.training.weight_decay,
    )

    # loss_fn：主分类损失函数。
    # 注意 pyvqnet 这里的参数顺序是 loss_fn(target, output)。
    loss_fn = nn.CrossEntropyLoss()

    # best_validation_accuracy：训练到目前为止出现过的最高验证准确率。
    best_validation_accuracy = 0.0

    # best_epoch：取得 best_accuracy 的 epoch 序号。
    best_epoch = 0

    # history：训练历史列表，每轮都会往里追加一条记录。
    history = []

    # complexity_score：当前方案编码复杂度得分。
    complexity_score = estimate_complexity_score(
        spec.model.encoding_gate_count,
        spec.model.encoding_depth,
    )

    if verbose:
        print(
            f"[Train] scheme={spec.model.scheme_name} | "
            f"gates={spec.model.encoding_gate_count} | "
            f"depth={spec.model.encoding_depth} | "
            f"complexity={complexity_score:.4f} | "
            f"train_size={x_train.shape[0]} | "
            f"validation_size={x_validation.shape[0]}"
        )

    # epoch：当前训练轮数。
    for epoch in range(1, spec.training.epochs + 1):
        # 切换到训练模式。
        model.train()

        # 下面这些变量用于累计这一整轮 epoch 的统计量。
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_align = 0.0
        sample_count = 0

        for x_batch, y_batch in data_generator(
            x_train,
            y_train,
            batch_size=spec.training.batch_size,
            shuffle=True,
        ):
            # x_tensor：当前 batch 的图像，转成 QTensor 才能进模型。
            x_tensor = QTensor(x_batch)

            # y_tensor：当前 batch 的标签，也转成 QTensor。
            y_tensor = QTensor(y_batch)

            # 前向传播：训练阶段需要辅助输出，所以 return_aux=True。
            logits, quantum_features, alignment_target = model(
                x_tensor,
                return_aux=True,
            )

            # ce_loss：主分类损失。
            ce_loss = loss_fn(y_tensor, logits)

            # align_loss：对齐正则损失。
            align_loss = compute_alignment_loss(quantum_features, alignment_target)

            # 如果当前方案没有对齐正则，就只使用交叉熵。
            if align_loss is None or spec.training.alignment_weight <= 0:
                total_loss = ce_loss
                align_value = 0.0
            else:
                total_loss = ce_loss + spec.training.alignment_weight * align_loss
                align_value = float(align_loss.item())

            # 清空上一轮 batch 的梯度。
            optimizer.zero_grad()

            # 反向传播，计算梯度。
            total_loss.backward()

            # 按优化器规则更新参数。
            optimizer.step()

            # current_batch_size：当前 batch 实际样本数。
            current_batch_size = int(y_batch.shape[0])

            # 累加这一轮的损失统计。
            epoch_loss += float(total_loss.item()) * current_batch_size
            epoch_ce += float(ce_loss.item()) * current_batch_size
            epoch_align += align_value * current_batch_size
            sample_count += current_batch_size

        # 每一轮结束后，重新统计训练集和内部验证集准确率。
        train_accuracy = evaluate_accuracy(
            model,
            x_train,
            y_train,
            batch_size=spec.training.batch_size,
        )
        validation_accuracy = evaluate_accuracy(
            model,
            x_validation,
            y_validation,
            batch_size=spec.training.batch_size,
        )

        # 下面三个值是这一整轮的平均损失。
        mean_loss = epoch_loss / max(sample_count, 1)
        mean_ce = epoch_ce / max(sample_count, 1)
        mean_align = epoch_align / max(sample_count, 1)

        # estimated_validation_total_score：
        # 按当前验证准确率估算参考分数，仅用于观察训练趋势。
        estimated_validation_total_score = estimate_total_score(
            validation_accuracy,
            spec.model.encoding_gate_count,
            spec.model.encoding_depth,
        )

        # 把这一轮结果记录进 history。
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_ce_loss": mean_ce,
                "train_align_loss": mean_align,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "estimated_validation_total_score": estimated_validation_total_score,
            }
        )

        # 如果本轮验证准确率不差于历史最好结果，就更新最优模型。
        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            save_parameters(model.state_dict(), str(model_path))

        if verbose:
            print(
                f"[Epoch {epoch:02d}] "
                f"loss={mean_loss:.4f} | "
                f"ce={mean_ce:.4f} | "
                f"align={mean_align:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_acc={validation_accuracy:.4f} | "
                f"best_val={best_validation_accuracy:.4f}"
            )

    # report：训练完成后的结构化总结。
    report = {
        "experiment_name": spec.experiment_name,
        "notes": spec.notes,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "train_size": int(x_train.shape[0]),
        "validation_size": int(x_validation.shape[0]),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "encoding_gate_count": spec.model.encoding_gate_count,
        "encoding_depth": spec.model.encoding_depth,
        "encoding_complexity_score": complexity_score,
        "estimated_validation_total_score": estimate_total_score(
            best_validation_accuracy,
            spec.model.encoding_gate_count,
            spec.model.encoding_depth,
        ),
        "model_config": asdict(spec.model),
        "training_config": asdict(spec.training),
        "history": history,
    }

    # 把报告写到磁盘。
    save_json(report_path, report)

    return report


def load_model_config_from_report(report_path: Path) -> ModelConfig:
    """
    从训练报告 JSON 中恢复模型结构配置。

    为什么需要这个函数？
    因为评估时我们希望尽量按训练时真正使用的结构恢复模型，
    而不是仅凭方案名做猜测。
    """

    # payload：整个 JSON 文件解析后的字典。
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    # 如果报告格式过旧，没有 model_config 字段，就抛出错误。
    if "model_config" not in payload:
        raise KeyError("model_config")

    # model_config：报告里保存的模型结构配置字典。
    model_config = payload["model_config"]

    # encoder_dims 在 JSON 中会被写成列表，
    # 这里恢复为 tuple，保证和 ModelConfig 的定义一致。
    model_config["encoder_dims"] = tuple(model_config["encoder_dims"])

    return ModelConfig(**model_config)


def evaluate_saved_model(
    root: Path,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    加载已经训练好的模型，并在测试集上重新评估。

    参数：
    - root：工作目录
    - output_dir：模型和报告所在目录，若为 None 则使用当前目录
    - verbose：是否打印详细信息
    """

    # 如果用户没指定目录，就使用该方案默认输出目录。
    if output_dir is None:
        output_dir = default_artifact_dir(root)

    # model_path：模型参数文件路径。
    model_path = output_dir / "best_model.vq"

    # report_path：训练报告路径。
    report_path = output_dir / "training_report.json"

    # result_dir：评估输出目录。
    # 这里主要用于保存逐样本预测表，避免把模型参数和运行结果混在一起。
    result_dir = default_output_dir(root)
    ensure_dir(result_dir)

    # prediction_table_path：逐样本预测表路径。
    prediction_table_path = result_dir / "test_predictions.csv"

    # 如果连模型文件都不存在，说明无法评估，直接报错。
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 优先尝试从训练报告中恢复模型配置。
    if report_path.exists():
        try:
            model_config = load_model_config_from_report(report_path)
        except KeyError:
            # 如果报告过旧或格式不完整，就退回到默认配置。
            model_config = get_default_spec().model
    else:
        model_config = get_default_spec().model

    # 评估阶段只读取测试集。
    x_test, y_test = load_test_dataset(root)

    # 按训练时同样的结构重新创建模型。
    model = QuantumImageClassifier(model_config)

    # 加载训练好的参数。
    model.load_state_dict(load_parameters(str(model_path)))

    # accuracy：最终重新评估得到的测试准确率。
    # prediction_rows：测试集每一张图片对应的预测结果。
    accuracy, prediction_rows = collect_prediction_rows(
        model,
        x_test,
        y_test,
        batch_size=128,
    )

    # result：评估结果总结字典。
    result = {
        "scheme_name": model_config.scheme_name,
        "display_name": model_config.display_name,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "prediction_table_path": str(prediction_table_path),
        "test_accuracy": accuracy,
        "encoding_gate_count": model_config.encoding_gate_count,
        "encoding_depth": model_config.encoding_depth,
        "encoding_complexity_score": estimate_complexity_score(
            model_config.encoding_gate_count,
            model_config.encoding_depth,
        ),
        "estimated_total_score": estimate_total_score(
            accuracy,
            model_config.encoding_gate_count,
            model_config.encoding_depth,
        ),
    }

    # 同时输出逐样本预测表，方便逐张检查测试集预测标签。
    actual_prediction_table_path = save_prediction_rows(
        prediction_table_path,
        prediction_rows,
    )
    result["prediction_table_path"] = str(actual_prediction_table_path)

    if verbose:
        print(f"Scheme: {result['display_name']}")
        print(f"Test accuracy: {result['test_accuracy']:.4f}")
        print(f"Encoding gate count: {result['encoding_gate_count']}")
        print(f"Encoding depth: {result['encoding_depth']}")
        print(
            f"Estimated complexity score: "
            f"{result['encoding_complexity_score']:.4f}"
        )
        print(f"Estimated total score: {result['estimated_total_score']:.4f}")
        print(f"Prediction table path: {result['prediction_table_path']}")

    return result


def format_comparison_markdown(results: list[dict], title: str) -> str:
    """
    把实验结果列表渲染成 Markdown 表格文本。

    参数：
    - results：多个实验结果组成的列表
    - title：Markdown 文档标题

    返回值：
    - 一个完整的 Markdown 字符串
    """

    # lines：逐行累积 Markdown 文本。
    lines = [
        f"# {title}",
        "",
        "| 实验名 | 方案 | 准确率 | 门数 | 深度 | 复杂度分 | 估算总分 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    # item：results 中的单条实验结果字典。
    for item in results:
        lines.append(
            "| {experiment_name} | {display_name} | {test_accuracy:.4f} | "
            "{encoding_gate_count} | {encoding_depth} | "
            "{encoding_complexity_score:.4f} | {estimated_total_score:.4f} |".format(
                **item
            )
        )

    # 最终把所有行拼起来，并在末尾补一个换行。
    return "\n".join(lines) + "\n"


