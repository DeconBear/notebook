# -*- coding: utf-8 -*-
"""路径五导论练习：实现「有钥匙才能开锁」规则。"""


def apply_unlock(state: dict) -> dict:
    """
    TODO: 若 has_key 为真且 door=='locked'，把门改为 'open'。
    否则原样返回（可先 copy）。
    """
    return None


if __name__ == "__main__":
    s1 = {"has_key": True, "door": "locked"}
    s2 = {"has_key": False, "door": "locked"}
    o1 = apply_unlock(s1)
    o2 = apply_unlock(s2)
    if o1 is None:
        print("请实现 apply_unlock")
    else:
        print("持钥匙:", o1, "期望 door=open")
        print("无钥匙:", o2, "期望 door=locked")
