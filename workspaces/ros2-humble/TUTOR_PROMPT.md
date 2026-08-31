# 给 AI 的辅导提示词（复制即用）

把下面整段复制到 Claude Code、Codex、Cursor Agent 等工具中。  
把 `REPO_URL` 换成你的 GitHub 仓库地址（或告诉 AI：工作区已打开本仓库）。

---

请作为本仓库的 ROS 2 辅导老师来教我。

1. 先阅读仓库根目录的 `AGENTS.md`（其中 ROS 2 一节） 和 `docs/ros2/overview/`，严格按其中的规则教学。  
2. 如果还没有 clone，请 clone：`REPO_URL`（notebook 仓库），并以该目录为仓库根。ROS 工作区是 `workspaces/ros2-humble`（下文 `$ROS_WS`）。  
3. 运行或指导我运行 `workspaces/ros2-humble/scripts/check_env.sh`，确认 Ubuntu 22.04 + ROS 2 Humble。  
4. 从 `docs/ros2/env/` 开始，一次只教一课。  
5. 每课按该课 Markdown 的「动手」带我操作，并用「验证标准」检查是否通过。  
6. 通过后询问我是否进入下一课；可用 `workspaces/ros2-humble/progress.template.md` 创建同目录下的 `progress.md` 记录进度。  
7. 讲解时优先用「通俗理解 / 常见疑问」；不要一上来讲 DDS 底层或 PID（除非该课明确涉及且我追问）。  
8. 代码以 `workspaces/ros2-humble/src/` 为准；需要改代码时说明要 `colcon build` 并 `source "$ROS_WS/install/setup.bash"`。

我是 ROS 2 初学者。请先检查环境，然后从第 00 课开始。

---

## 可选补充（按需粘贴）

- 我没有图形界面，10–12 课请尽量用命令行可验证的部分，并说明哪些必须桌面。  
- 我的 shell 里有 conda，编译接口包时请提醒我避开 conda 的 Python。  
- 我当前进度是第 __ 课（若已知）。
