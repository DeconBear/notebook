# 依次渲染机器人各章动画（在 _manim 目录运行）
$ErrorActionPreference = "Stop"
$env:Path = "C:\Users\qzx\AppData\Local\Microsoft\WinGet\Links;C:\Users\qzx\AppData\Local\Programs\ffmpeg\ffmpeg-8.1.2-essentials_build\bin;" + $env:Path
$jobs = @(
  @("cspace_workspace.py", "CspaceWorkspace"),
  @("trajectory_ptp.py", "JointTrajectory"),
  @("dh_3r.py", "DH3R"),
  @("so3_compose.py", "RotationOrder"),
  @("fourbar.py", "FourBar"),
  @("se2_screw.py", "SE2Screw")
)
foreach ($j in $jobs) {
  Write-Host "==== $($j[0]) $($j[1]) ===="
  python -m manim -qm $j[0] $j[1]
  if ($LASTEXITCODE -ne 0) { throw "render failed: $($j[0])" }
}
Write-Host "ALL OK"
