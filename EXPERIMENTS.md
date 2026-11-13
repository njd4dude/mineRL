# Experiment log

Results from training the treechop agent. Each evaluation is saved in the run folder of every model
it scored: `runs/<kind>/<run>/evals/<date>_<time>/results.json` (logs per seed, `died_at_step`,
`won_at_step`, the other models scored on the same worlds) plus game videos with `--record`
(`final_seed3.mp4`, ...). Random actions: `runs/random/evals/`. Below, `evals/<date>_<time>` means
that folder in the model's own run folder.

## Setup

- Env `MineRLTreechopAxe320-v0`: 320x180 render, agent sees 64x64 (HUD cropped), frame skip 4,
  attack always held, starts with an iron axe, time frozen, mobs on.
- Actions `[move, yaw, pitch]` = 3 x 11 x 9 (MultiDiscrete); move = stand / forward / forward+jump.
- Reward +1 per log. Episode ends at 64 logs, on death (axe count drops to 0), or after 2,000 steps.
- Eval: `evaluate.py`, seeds 1-24, 1,000 steps per world, same seeds for every model.
- Networks (`networks.py`, `--arch`): NatureCNN (SB3 default, 0.61M params) or ImpalaCNN (ResNet,
  channels 32/64/64, 2 residual blocks per stage, 2.56M params).

## Run names

`runs/<kind>/model_<date>_<time>_<name>`, kind = `bc` (bc_train.py), `ppo` (train_treechop.py) or
`test` (names starting with `test`).

- `bc_<data>`: data as `<version>-<segments>` per folder, e.g. `bc_6xx-1762_8xx-1957`. `bc_impala_...`
  = ImpalaCNN. `_plusN` = continued N more epochs with `--init`; `_cont` = continued after a stop.
- `ppo_<start>_<setting>`: PPO from a BC model (`ppo_bc_6xx-304_kl05` = from `bc_6xx-304`,
  target_kl 0.05) or from scratch (`ppo_scratch_...`).
- Old names (`bc_multi`, `bc_all`, `finetune_kl05`, ...) are mapped in `runs/renamed_runs.json`.

## Results: 24 worlds, 1,000 steps

BC: 20 epochs, lr 3e-4 cosine, 5 s window before each log break, unless noted.
PPO fine-tunes: lr 1e-4, ent 0.001, target_kl 0.05, 5 value warm-up updates, 120k steps, 4 envs.

| policy | mean | median | worlds at 64 | 0-log games | deaths | eval |
| --- | --- | --- | --- | --- | --- | --- |
| PPO `ppo_bc_6xx-1762_8xx-1957_plus20_kl05` | **59.8** | 64 | 19 | 0 | 0 | `20261111_191652` |
| PPO `ppo_bc_impala_all_kl05` | **59.6** | 64 | 17 | 0 | 1 | `20261112_160817` |
| BC `bc_impala_..._10xx-777_cont` (ImpalaCNN, all versions) | **58.8** | 64 | 14 | 0 | 0 | `20261112_111541` |
| BC `bc_impala_6xx-1763_8xx-1971` (ImpalaCNN, 6.x + 8.x) | 51.1 | 64 | 13 | 0 | 2 | `20261112_124707` |
| BC `bc_6xx-1762_8xx-1957_plus20` (40 epochs) | 42.8 | 44.5 | 4 | 0 | 3 | `20261111_142003` |
| BC `bc_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777` (all versions, 7,599 segments) | 41.6 | 42 | 6 | 0 | 1 | `20261111_211618` |
| BC `bc_6xx-1762_8xx-1957` (3,719 segments, 1.13M examples) | 38.3 | 38.5 | 7 | 2 | 0 | `20261111_131528` |
| BC `bc_6xx-1762_8xx-1957_plus40` (60 epochs) | 37.5 | 36 | 7 | 1 | 1 | `20261111_145652` |
| PPO `ppo_bc_6xx-304_kl05` (120k checkpoint) | 34.3 | 30.5 | 3 | 2 | 1 | `20261110_193136` + `_220203` |
| BC `bc_6xx-1603` | 26.5 | 22 | 4 | 0 | 4 | `20261111_105730` |
| BC `bc_6xx-304` | 18.7 | 15.5 | 2 | 3 | 6 (drowning) | `20261110_114949` |
| PPO from scratch `ppo_scratch_30k` (30k steps) | 3.0 | 1.5 | 0 | 8 | 6 | `20261111_155924` |
| random actions | 2.5 | 1.5 | 0 | 6 | 1 | `runs/random/evals/20261111_155924` |

Scores are **logs collected per game** (1,000 steps, max 64), averaged over the 24 worlds.

**How to read a comparison** like "+7.9 logs (SE 2.9)": on each of the 24 worlds, take the new
model's logs minus the old model's logs, and average. +7.9 means the new model got 7.9 more logs per
game on average. SE (standard error) is how much that average could move from luck alone (spread of
the 24 differences / sqrt(24)). If the difference is more than about 2 x SE it is very likely real;
if less, it could be luck. Each comparison below names the model it is measured against.

### Steps to reach 64 logs (worlds that got there)

`won_at_step`, recorded since Nov 12; for older evals recovered from video frame counts (+-1 step).

| model | worlds at 64 | median | mean | fastest | slowest |
| --- | --- | --- | --- | --- | --- |
| BC NatureCNN `_plus20` | 4 | 800 | 788 | 589 | 964 |
| BC ImpalaCNN all versions | 14 | 789 | 762 | 401 | 1000 |
| PPO on NatureCNN `_plus20` | 19 | 756 | 746 | 538 | 969 |
| PPO on ImpalaCNN all versions | 17 | 667 | 695 | 538 | 876 |

On shared wins, PPO ImpalaCNN is 51 steps faster than PPO NatureCNN (SE 32, 14 worlds) and 54 faster
than its BC start (SE 50, 10 worlds): probably the best model, not yet clearly.

## What we learned

**Baselines.** Random actions: 2.5 logs per game. PPO from scratch (30k steps, no BC): 3.0 logs per
game, +0.5 logs vs random (SE 0.9), so no better than random. With only +1 per log and no help
from demonstrations, PPO from scratch learns almost nothing at this budget.

**More BC data (small network, NatureCNN, 20 epochs).** Each step adds data and is compared with
the step before it:

| step | from | to | change (logs per game) | SE | real? |
| --- | --- | --- | --- | --- | --- |
| more 6.x clips | `bc_6xx-304`: 304 clips, 18.7 logs | `bc_6xx-1603`: 1,603 clips, 26.5 logs | +7.9 | 2.9 | yes |
| add 8.x clips | `bc_6xx-1603`: 26.5 logs | `bc_6xx-1762_8xx-1957`: 3,719 clips, 38.3 logs | +11.8 | 3.8 | yes |
| add 7.x, 9.x, 10.x | `bc_6xx-1762_8xx-1957`: 38.3 logs | all versions: 7,599 clips, 41.6 logs | +3.3 | 4.1 | could be luck |

So with the small network, more data helped up to about 3,700 clips and then stopped. All
versions vs the 40-epoch 6.x + 8.x model (`_plus20`, 42.8 logs): -1.2 logs (SE 3.0), the same.

The same filter is used for every version: a recording is kept only if the player broke at least 5
logs with an axe; a log break is dropped if it took more than 2 s of attacking (probably by hand)
or came after the player placed logs (taking their own building apart). From the kept recordings we
train only on the 5 s before each log break.

**More epochs (NatureCNN, 6.x + 8.x).** 20 / 40 / 60 epochs: 38.3 / 42.8 / 37.5 logs per game.
40 vs 20 epochs: +4.5 logs (SE 3.8); 60 vs 40: -5.3 logs (SE 4.0); both could be luck. Validation
loss barely moved (3.126 -> 3.116 -> 3.111) while the gap to the training loss grew (0.013 -> 0.028
-> 0.039): the model was starting to memorise. 20 epochs is enough.

**A bigger network was the limit.** Same all-versions data, only the network changed:

| comparison | from | to | change (logs per game) | SE | real? |
| --- | --- | --- | --- | --- | --- |
| bigger network, all versions | NatureCNN: 41.6 logs | ImpalaCNN: 58.8 logs | +17.1 | 4.4 | yes |
| bigger network, 6.x + 8.x | NatureCNN 40 epochs (`_plus20`): 42.8 logs | ImpalaCNN: 51.1 logs | +8.3 | 2.9 | yes |
| more data, big network | ImpalaCNN 6.x + 8.x: 51.1 logs | ImpalaCNN all versions: 58.8 logs | +7.6 | 4.6 | probably |

With the big network, the extra versions do help: the 6.x + 8.x ImpalaCNN had a few very bad games
(6-24 logs on seeds 5, 9, 11, 21, 22, where the all-versions one got 64), and on the same held-out
6.x / 8.x clips the all-versions model predicts the human actions better (validation loss 2.953 vs
2.973 and 2.973 vs 3.002). Validation loss overall: ImpalaCNN 3.046 vs NatureCNN 3.198, lower on
every version. ImpalaCNN takes about 4.5 min per epoch on all versions (NatureCNN about 3 min).

- The first ImpalaCNN try learned nothing (it only ever predicted the most common action;
  validation loss stuck at 3.948): SB3 re-initialises every conv layer (orthogonal, gain sqrt 2),
  the activations grew about 20x through the residual blocks, and about 3/4 of the features died
  in the first updates. Fixed with a learned residual scale that starts at 0 (ReZero). Failed run:
  `model_20261111_224940_bc_impala_...`. The fixed run stopped after epoch 4
  (`model_20261112_000640_...`) and was continued for epochs 5-20 (lr 2.7e-4 -> 0) as `..._cont`.

**PPO fine-tuning** (120k steps, about 2.5-3 h). Each fine-tune is compared with the BC model it
started from:

| fine-tune | BC start | after PPO | change (logs per game) | SE | real? |
| --- | --- | --- | --- | --- | --- |
| `ppo_bc_6xx-304_kl05` | `bc_6xx-304`: 18.7 logs | 34.3 logs | +15.7 | 3 | yes |
| `ppo_bc_6xx-1762_8xx-1957_plus20_kl05` | `_plus20`: 42.8 logs | 59.8 logs | +17.0 | 4.0 | yes |
| `ppo_bc_impala_all_kl05` | ImpalaCNN all versions: 58.8 logs | 59.6 logs | +0.8 | 2.0 | can't tell |

On the ImpalaCNN model the test can't show a gain, because the BC model already reached 64 logs on
14 of 24 worlds. Training says PPO did help: logs per 2,000-step training game rose from about 46
to 58.6 (the NatureCNN fine-tune ended at 55.7 after the same steps), and it reaches 64 logs about
50 steps sooner (table above). PPO's result depends mostly on where it starts: it reinforces what
the BC model already does some of the time.

- target_kl 0.02 was too strict (every update stopped in its first epoch); 0.05 works.
- `ppo_bc_6xx-304_kl05` stopped improving: about 38 logs per training game from 126k to 163k steps.
- Training logs per game (`ep_rew_mean`) can't be compared directly with eval scores: training
  games are 2,000 steps and use random worlds.

**The 1,000-step test is now near its limit** (14-19 of 24 worlds at 64 logs for the best models).
Every world has been cleared by some model, and different models fail on different worlds, so the
remaining misses are mostly bad luck. To compare the best models: use steps to reach 64, and play
each world 2-3 times.

## Failure modes (from eval videos)

- NatureCNN BC: drowning (all 6 deaths of `bc_6xx-304`), staring at the sky inside leaves, worst in
  plains or near water.
- After PPO: drowning gone; left: digging into the ground (attack is held, so looking down digs),
  wandering into dark caves (seed 5 death, likely a mob), getting stuck in hilly terrain by water.
- ImpalaCNN BC and its fine-tune both do worst on seed 1 (28 and 30 logs).

## Evaluation noise

- The same model on the same world isn't repeatable (BC on seed 3: 64, 64, 16, 14), and Minecraft
  itself isn't deterministic (frames diverge by tick ~20 with the same actions).
- So: 24 worlds, paired differences with SE, and treat a few logs as noise.

## Infrastructure

- Each Minecraft instance uses ~3-4 GB RAM at `--mc-mem 2G`; 4 instances is the limit in 27 GB WSL.
  Cursor's Pyright language server can grow to 8-12 GB; check before 4-instance runs.
- About 2 Minecraft restarts per hour of PPO training. Causes: all 4 starting at once (startup
  timeouts), and Java OutOfMemoryError at 2G after many world resets (an update can stall ~20 min).
  Options: `--mc-mem 3G` (~24 GB total), staggered startup, planned restarts every N games.
- MineRL returns a random observation on a timeout (caused fake "logs" once); now caught via
  `info["error"]` and the instance is restarted. `ObservationFromLifeStats` hung with 4 instances,
  so death is detected from the axe count.
- PPO speed: ~17-20 agent steps/s in total over 4 instances (SB3 `time/fps` is the sum), incl. PPO
  updates and resets; ~2.5-3 h per 120k steps.

## Earlier evals (before the current setup)

8 worlds, episodes didn't end on death: random 2.9-3.2; BC v1 inventory filter
(`bc_6xx-303_v1-inventory-filter`) 12.4; v2 mine_block filter + window (`bc_6xx-304_v2-7actions`)
17.9; v3 move/yaw/pitch actions (`bc_6xx-304`) 21.4 / 26.8; `ppo_bc_6xx-304_kl02_no-death-end` 21.4
at 30k. 4 worlds (`bc_6xx-304`'s `evals/20261110_162445`, `_192334`): BC 25.2, `ppo_bc_6xx-304_kl05`
47.8-53.2. Short 50-80 step and single-seed evals in `bc_6xx-304`'s `evals/` were video tests.

## Next steps

- Make the test discriminate again: compare steps to 64, play each world 2-3 times.
- Longer PPO on ImpalaCNN BC (300-500k steps, overnight) with `--mc-mem 3G` and staggered startup.
- Look at the seed 1 videos (both ImpalaCNN models fail there).
- Off-policy demo-based RL (DQfD / SQIL) or a faster simulator (CraftGround trial) to afford more RL.
