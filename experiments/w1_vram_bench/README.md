# W1 VRAM / speed benchmark

Task `Isaac-Velocity-Rough-Unitree-Go2-v0`, default PPO config (24 steps/env/iter), seed 0, 30 iterations, headless.
GPU: RTX 4070 Laptop 8 GB, platform profile `balanced`, **GPU power limit 55 W** (hardware max 140 W).

Run with: `isaac/train_policy.sh Isaac-Velocity-Rough-Unitree-Go2-v0 <N> 0 30 w1_vram_bench/n<N>`

| num_envs | peak VRAM (MiB) | steps/s | iter time | est. 1500 iters |
|---------:|----------------:|--------:|----------:|----------------:|
| 1024     | 3932            | 10.1k   | 2.4 s     | ~1.0 h          |
| 2048     | 4544            | 11.7k   | 4.2 s     | ~1.8 h          |
| 4096     | 5940            | 12.1k   | 8.1 s     | ~3.4 h          |

Peak VRAM is the max over the whole process, so it includes startup and terrain generation.

## Findings
- 4096 envs (the Isaac Lab default) fits, with about 2.2 GB of headroom.
- Throughput levels off at about 12k steps/s. Rollout collection (physics) takes over 95% of each iteration;
  the PPO update is negligible.
- It is GPU-bound: GPU utilization is about 95% while the power draw stays at the 55 W cap (around 1.9 GHz SM clock),
  and the CPU sits mostly idle. Switching the laptop's power profile to `performance` should raise the limit
  and speed things up.
- Measured with `performance` (`throttle_thermal_policy=1`): 4096 envs gave 12.3k steps/s and the GPU stayed
  at 55 W, so the ACPI profile does not lift the GPU cap. NVIDIA's Dynamic Boost daemon `nvidia-powerd`
  is installed (`/usr/bin/nvidia-powerd`), but its systemd service has never been installed or enabled.
  That is the likely reason the GPU sits at its base power limit.

## After enabling `nvidia-powerd` (GPU power limit 55 W → 140 W)
| num_envs | peak VRAM (MiB) | steps/s | iter time | est. 1500 iters |
|---------:|----------------:|--------:|----------:|----------------:|
| 4096     | 5940            | 13.7k   | 7.2 s     | ~3.0 h          |

Only about 12% faster. The GPU now runs at its maximum boost clock (2565 MHz) and draws about 83 W at about 95% utilization,
so it is limited by clock speed, not power. That is the ceiling for rough terrain on this laptop. For comparison, the flat task runs at about 110k steps/s.

**Decision:** use 4096 envs for the rough baseline.
