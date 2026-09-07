# 批量渲染适合视频讲解的短片（在 docs/_manim 运行；雅可比在 robotics/_manim）
$ErrorActionPreference = "Stop"
$env:Path = "C:\Users\qzx\AppData\Local\Microsoft\WinGet\Links;C:\Users\qzx\AppData\Local\Programs\ffmpeg\ffmpeg-8.1.2-essentials_build\bin;" + $env:Path
$here = $PSScriptRoot
$jobs = @(
  @("pid_step.py", "PIDStep"),
  @("zeta_step.py", "ZetaStep"),
  @("root_locus.py", "RootLocusWalk"),
  @("bode_sine.py", "BodeSine"),
  @("kalman_1d.py", "KalmanBand"),
  @("lqr_cart.py", "LQRCart"),
  @("eigen_stretch.py", "EigenStretch"),
  @("gd_bowl.py", "GradDesc"),
  @("secant_tangent.py", "SecantTangent"),
  @("neuron_spike.py", "NeuronSpike"),
  @("stdp_pair.py", "STDPPair"),
  @("bloch_h.py", "BlochH"),
  @("residual_morph.py", "ResidualMorph")
)
Set-Location $here
foreach ($j in $jobs) {
  Write-Host "==== $($j[0]) $($j[1]) ===="
  python -m manim -qm $j[0] $j[1]
  if ($LASTEXITCODE -ne 0) { throw "render failed: $($j[0])" }
}
Write-Host "==== jacobian ===="
Set-Location (Join-Path $here "..\robotics\_manim")
python -m manim -qm jacobian_sing.py JacobianSing
if ($LASTEXITCODE -ne 0) { throw "render failed: jacobian" }

$root = Split-Path (Split-Path $here -Parent) -Parent
# $here is docs/_manim ; repo is parents[1]
$docs = Split-Path $here -Parent
function Copy-Clip($scene, $cls, $dest) {
  $src = Join-Path $here "media\videos\$($scene)\720p30\$cls.mp4"
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  Copy-Item $src $dest -Force
  Write-Host " -> $dest"
}
Copy-Clip pid_step PIDStep (Join-Path $docs "control\classical\pid\images\pid_anim.mp4")
Copy-Clip zeta_step ZetaStep (Join-Path $docs "control\classical\transfer\images\zeta_anim.mp4")
Copy-Clip root_locus RootLocusWalk (Join-Path $docs "control\classical\root-locus\images\locus_walk.mp4")
Copy-Clip bode_sine BodeSine (Join-Path $docs "control\classical\frequency\images\bode_sine.mp4")
Copy-Clip kalman_1d KalmanBand (Join-Path $docs "control\modern\kalman\images\kf_anim.mp4")
Copy-Clip lqr_cart LQRCart (Join-Path $docs "control\modern\lqr\images\lqr_cart.mp4")
Copy-Clip eigen_stretch EigenStretch (Join-Path $docs "math\eigen\images\eigen_anim.mp4")
Copy-Clip gd_bowl GradDesc (Join-Path $docs "math\optimization\images\gd_anim.mp4")
Copy-Clip secant_tangent SecantTangent (Join-Path $docs "math\derivative\images\secant_anim.mp4")
Copy-Clip neuron_spike NeuronSpike (Join-Path $docs "neuro\hh-lif\images\spike_anim.mp4")
Copy-Clip stdp_pair STDPPair (Join-Path $docs "neuro\stdp\images\stdp_anim.mp4")
Copy-Clip bloch_h BlochH (Join-Path $docs "quantum\computing\images\bloch_h.mp4")
Copy-Clip residual_morph ResidualMorph (Join-Path $docs "science\overview\images\residual_anim.mp4")
$jac = Join-Path $docs "robotics\_manim\media\videos\jacobian_sing\720p30\JacobianSing.mp4"
Copy-Item $jac (Join-Path $docs "robotics\kinematics\images\jacobian.mp4") -Force
Write-Host "ALL OK"
