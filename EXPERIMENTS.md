# Experiment log

Results and observations from training the treechop agent. Raw evaluation results are in
`runs/evals/<date>_<time>.json` (logs per seed per model), with videos in `<date>_<time>_videos/`
when `--record` was used.

## Setup (current)

- Env: `MineRLTreechopAxe320-v0`, 320x180 render, the agent sees 64x64 (HUD cropped), frame skip 4.
  Attack is always held.
- Actions: `[move, yaw, pitch]` = 3 x 11 x 9 (MultiDiscrete). Move: stand / forward / forward+jump.
- Reward: +1 per log collected. The episode ends at 64 logs or on death (the axe count drops to 0).
- Evaluation: `evaluate.py`, 1000 agent steps per world, the same seeds for every model.

## Results

### Main baseline: 24 worlds (seeds 1-24), episodes end on death

| policy | mean | median | wins (64) | 0-log games | deaths | file |
|---|---|---|---|---|---|---|
| BC `bc_multi` | **18.7** | 15.5 | 2 | 3 | 6/24 (all drowning) | `20261110_114949.json` |

Still to run: random actions on the same 24 worlds.

### Earlier: 8 worlds (seeds 1-8), before episodes ended on death

| policy | mean | median | wins | file |
|---|---|---|---|---|
| random actions | 2.9 / 3.2 (two runs) | 2.5 | 0 | `20261109_221254`, `20261109_225115` |
| PPO from scratch `cam5` (simple actions) | 4.6 | 4.0 | 0 | `20261109_221254` |
| BC v1 `bc` (inventory filter) | 12.4 | 5.5 | 0 | `20261109_221254` |
| BC v2 `bc_breaks` (mine_block stats + window before a log break) | 17.9 | 10.5 | 0 | `20261109_221942` |
| BC v3 `bc_multi` (move/yaw/pitch actions) | 21.4, re-run 26.8 | 13 / 20 | 1 / 2 | `20261109_225115`, `20261110_100622` |
| fine-tune `finetune`, checkpoint at 30k | 21.4 | 11.5 | 2 | `20261109_234542` |
| fine-tune `finetune`, checkpoint at 40k | 17.6 | 14.5 | 1 | `20261109_234542` |

(The short runs in `runs/evals/`, of 50-80 steps or a single seed, were video/debug tests. A run that
showed 3,259 fake "logs" from the Minecraft-hang bug was deleted.)

### Quick check: 4 worlds (seeds 1-4), episodes end on death, `20261110_162445.json`

| policy | s1 | s2 | s3 | s4 | mean | deaths |
|---|---|---|---|---|---|---|
| random | 0 | 2 | 11 | 7 | 5.0 | 0/4 |
| BC `bc_multi` | 4 | 17 | 16 | 64 | 25.2 | 0/4 |
| `finetune_kl05` checkpoint at 50k | 16 | 34 | 64 | 64 | 44.5 | 0/4 |
| `finetune_kl05` final (~120k) | 31 | 39 | 60 | 61 | **47.8** | 0/4 |

BC on these seeds across all 4 evaluations so far: s1 9/0/1/4, s2 26/21/31/17, s3 64/64/32/16,
s4 32/64/64/64 (about 32 on average). Both fine-tuned checkpoints beat every BC run on seeds 1 and 2,
so this is the first real sign that fine-tuning helps. Only 4 worlds, though: confirm on 24.

## PPO fine-tuning runs (from `bc_multi`)

All runs: lr 1e-4, ent_coef 0.001, n_steps 1024 x 4 envs = 4,096 steps per update, 10 epochs,
batch 64, 5 value warm-up updates (only the value head trains).

| run | target_kl | steps | result |
|---|---|---|---|
| `finetune_death` | 0.02 | ~94k (stopped) | Every update early-stopped at epoch 0 (KL ~0.04 > 0.03 limit). ep_rew_mean flat at ~21-24; explained_variance rose to ~0.6-0.75. No forgetting, but hardly any learning. |
| `finetune_kl05` | 0.05 | ~120k (paused, can `--resume`) | Most updates complete several epochs. At ~100k: ep_rew_mean rose from ~20 to ~34, explained_variance ~0.6-0.8, approx_kl ~0.03-0.046, clip_fraction ~0.3-0.4 (high), entropy 4.1 -> 4.65 (max is 5.69). Needs the 24-world eval. |

Note: training ep_rew_mean is **not comparable** to eval scores. Training episodes average about
1,500 steps (eval: 1,000), use random worlds, and average over the last 100 episodes from many
policy versions.

## Observations

### Data / behaviour cloning
- Filtering VPT clips by inventory kept clips of people just walking around. Filtering with the
  `mine_block` stats, plus a window before each log break, fixed it (BC 12.4 -> 17.9).
- Matching the action space to what humans actually do (separate move, yaw and pitch steps)
  helped further (about 21 on 8 worlds).
- BC was the big jump overall: random about 3, PPO from scratch about 5, BC about 19-21.

### Failure modes (from eval videos)
- **Drowning**: all 6 deaths in the 24-world baseline. The agent walks into water and keeps
  chopping and holding forward.
- **Staring at the sky** inside tree leaves.
- Does best in dense forests, worst in plains or near water (0-log games).

### Evaluation noise
- The same model on the same world is not repeatable: BC on seed 3 scored 64, 64, 16 and 14
  across runs. The same BC model on 8 worlds averaged 21.4 one time and 26.8 the next.
- Minecraft itself isn't deterministic: the same random action sequence on world 3 gave 11 and
  10 logs, and the frames diverge by about tick 20.
- So use 24+ worlds, and treat differences of a few logs as noise.

### PPO behaviour
- target_kl 0.02 was too strict for this setup: the first epoch alone already exceeded
  1.5 x 0.02 = 0.03, so each update used its data for less than one pass.
- During warm-up, approx_kl was about 3e-8, not 0: GPU rounding (different batch sizes, TF32),
  not a real change.
- PPO is on-policy: each 4,096-step batch is used for one update and thrown away, which is
  expensive when each step costs about 50 ms of Minecraft. If PPO stays slow, try an off-policy
  method that also uses the demos (e.g. SQIL or DQfD-style replay buffers; this approach won the
  MineRL competitions).

### Infrastructure
- Each Minecraft instance uses about 3.3 GB of RAM (Java about 2.4 GB + worker about 0.85 GB);
  4 instances is the practical limit in 28 GB WSL.
- On a timeout, MineRL returns a random observation (that caused the fake logs). It's now caught
  via `info["error"]` and the instance is restarted.
- `ObservationFromLifeStats` caused hangs with 4 instances; death is detected from the axe
  count instead.

## Next steps
- Confirm `finetune_kl05` on the 24 worlds (the 4-world check: 47.8 vs BC 25.2).
- Resume `finetune_kl05` toward 200k; ep_rew_mean was flat at ~34 from ~98k-120k.
- If fine-tuning loses BC skills: add a KL penalty toward the BC policy.
- If the reward collapses: lower target_kl or the learning rate (clip_fraction is already high).
