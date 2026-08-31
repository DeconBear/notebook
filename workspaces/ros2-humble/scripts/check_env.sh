#!/usr/bin/env bash
# 检查本教程所需的基本环境（不修改系统）
set -euo pipefail

ok=0
warn=0

pass() { echo "[OK] $*"; ok=$((ok+1)); }
fail() { echo "[FAIL] $*"; }
note() { echo "[!!] $*"; warn=$((warn+1)); }

echo "== ROS 2 Humble Notes: environment check =="

if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  . /etc/os-release
  if [[ "${VERSION_ID:-}" == "22.04" ]]; then
    pass "Ubuntu 22.04 detected ($PRETTY_NAME)"
  else
    note "Ubuntu is '${VERSION_ID:-unknown}' (tutorial targets 22.04 / Humble)"
  fi
else
  note "Cannot read /etc/os-release"
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  pass "Found /opt/ros/humble/setup.bash"
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
else
  fail "ROS 2 Humble not found at /opt/ros/humble"
  echo "    Install: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html"
fi

if command -v ros2 >/dev/null 2>&1; then
  pass "ros2 on PATH (ROS_DISTRO=${ROS_DISTRO:-unset})"
else
  fail "ros2 command not available (source Humble setup.bash first)"
fi

if command -v colcon >/dev/null 2>&1; then
  pass "colcon available"
else
  note "colcon not found (sudo apt install python3-colcon-common-extensions)"
fi

py=$(command -v python3 || true)
if [[ -n "$py" ]]; then
  if [[ "$py" == /usr/bin/python3 ]]; then
    pass "python3 is $py"
  else
    note "python3 is $py (conda/other). For interface builds prefer /usr/bin/python3"
  fi
fi

if dpkg -l 'ros-humble-ros-gz' 2>/dev/null | grep -q '^ii'; then
  pass "ros-humble-ros-gz installed (lessons 11-12)"
else
  note "ros-humble-ros-gz not installed (needed for Gazebo lessons 11-12)"
fi

if [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  pass "Graphical display detected (RViz/Gazebo OK to try)"
else
  note "No DISPLAY/WAYLAND_DISPLAY (lessons 10-12 GUI may fail)"
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -d "$REPO_ROOT/src/py_pubsub" ]]; then
  pass "Repository layout looks good ($REPO_ROOT)"
else
  fail "Unexpected repo layout under $REPO_ROOT"
fi

echo
echo "Done. Fix any [FAIL] before studying; [!!] are warnings."
