# -*- coding: utf-8 -*-
"""
路径五导论 demo：规则世界模型 vs 死记转移表
==========================================
同一把「钥匙—门」微型世界：
  - SymbolicWM：显式规则（有钥匙才能开门）
  - LookupWM：只记忆训练时见过的 (s, a) → s'
换一把没在训练集里出现的钥匙，查找表会失败，规则仍然成立。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, "..", "images")
os.makedirs(_IMAGES_DIR, exist_ok=True)


def step_rules(state: dict, action: str) -> dict:
    """可执行符号动力学。state: loc, has_key, door, key_id。"""
    s = dict(state)
    loc, has_key, door, key_id = s["loc"], s["has_key"], s["door"], s["key_id"]
    if action == "pickup" and loc == "room" and not has_key:
        s["has_key"] = True
    elif action == "unlock":
        # 定律：拿着匹配的钥匙才能开锁（这里任意非空 key_id 都匹配）
        if has_key and key_id and door == "locked":
            s["door"] = "open"
    elif action == "go_out":
        if door == "open":
            s["loc"] = "outside"
    elif action == "go_in":
        if s["loc"] == "outside":
            s["loc"] = "room"
    return s


def train_lookup(episodes: list[tuple[dict, str]]) -> dict:
    """死记 (冻结后的状态元组, 动作) → 下一状态。"""
    table = {}
    for s, a in episodes:
        key = (s["loc"], s["has_key"], s["door"], s["key_id"], a)
        table[key] = step_rules(s, a)
    return table


def step_lookup(table: dict, state: dict, action: str) -> dict:
    key = (state["loc"], state["has_key"], state["door"], state["key_id"], action)
    if key in table:
        return dict(table[key])
    # 没见过：保持原状（幻觉成「什么都没发生」）
    return dict(state)


def rollout(step_fn, start: dict, actions: list[str]) -> list[dict]:
    s = dict(start)
    traj = [dict(s)]
    for a in actions:
        s = step_fn(s, a)
        traj.append(dict(s))
    return traj


def goal_reached(traj: list[dict]) -> bool:
    return traj[-1]["loc"] == "outside" and traj[-1]["door"] == "open"


def plot_compare(ok_rule: list[bool], ok_lookup: list[bool], labels: list[str],
                 save_name: str = "symbolic_vs_lookup.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [int(v) for v in ok_rule], width=w,
           label="符号规则 Exec(Rules)", color="#2E86AB")
    ax.bar([i + w / 2 for i in x], [int(v) for v in ok_lookup], width=w,
           label="查找表（只记训练转移）", color="#C1666B")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("任务成功（出门且门开）")
    ax.set_title("同一动力学：规则组合泛化，死记在新钥匙上失败")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_IMAGES_DIR, save_name), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {save_name}")


def main() -> None:
    print("=" * 56)
    print("路径五 · 规则世界模型 vs 死记转移")
    print("=" * 56)
    plan = ["pickup", "unlock", "go_out"]
    train_start = {"loc": "room", "has_key": False, "door": "locked", "key_id": "iron"}
    # 训练：只见过 iron 钥匙
    train_pairs = []
    s = dict(train_start)
    for a in plan:
        train_pairs.append((dict(s), a))
        s = step_rules(s, a)
    table = train_lookup(train_pairs)

    tests = [
        ("训练钥匙 iron", {"loc": "room", "has_key": False, "door": "locked", "key_id": "iron"}),
        ("新钥匙 gold", {"loc": "room", "has_key": False, "door": "locked", "key_id": "gold"}),
        ("已持钥匙直接开", {"loc": "room", "has_key": True, "door": "locked", "key_id": "gold"}),
    ]
    ok_r, ok_l, labels = [], [], []
    for name, start in tests:
        tr = rollout(step_rules, start, plan)
        tl = rollout(lambda st, a: step_lookup(table, st, a), start, plan)
        print(f"\n[{name}]")
        print(f"  规则轨迹终点: {tr[-1]}")
        print(f"  查找表终点:   {tl[-1]}")
        print(f"  规则成功={goal_reached(tr)}  查找表成功={goal_reached(tl)}")
        ok_r.append(goal_reached(tr))
        ok_l.append(goal_reached(tl))
        labels.append(name)

    plot_compare(ok_r, ok_l, labels)
    print(f"\n完成。图片在 {_IMAGES_DIR}")
    print("查找表在训练钥匙上可以「装对」，换 key_id 就不再触发开门。")


if __name__ == "__main__":
    main()
