# Experiment log

Results and observations from training the treechop agent. Each evaluation is saved in the run
folder of every model it scored, as `runs/<kind>/<run>/evals/<date>_<time>/results.json` (logs per
seed, plus which other models were scored on the same worlds), with the game videos next to it when
`--record` was used (`final_seed3.mp4`, ...). Random actions: `runs/random/evals/`. Below, an eval is
referred to by its `<date>_<time>`, e.g. `evals/20261110_114949` in that model's run folder.

## Setup (current)

- Env: `MineRLTreechopAxe320-v0`, 320x180 render, the agent sees 64x64 (HUD cropped), frame skip 4.
Attack is always held.
- Actions: `[move, yaw, pitch]` = 3 x 11 x 9 (MultiDiscrete). Move: stand / forward / forward+jump.
- Reward: +1 per log collected. The episode ends at 64 logs or on death (the axe count drops to 0).
- Evaluation: `evaluate.py`, 1000 agent steps per world, the same seeds for every model.



## Run names

Run folders are `runs/<kind>/model_<date>_<time>_<name>`: `runs/bc/` for behaviour cloning
(bc_train.py), `runs/ppo/` for PPO (train_treechop.py), `runs/test/` for any run whose name starts
with `test`. Each run's evaluations are in its own `evals/` folder (see above); VPT reviews and
surveys are in `runs/vpt_review/` and `runs/vpt_survey/`.
`<name>` says what the run is:

- `bc_<data>`: behaviour cloning on VPT data, `<version>-<segments>` per data folder, e.g.
`bc_6xx-1762_8xx-1957` = 1,762 segments from data/vpt_6xx (treechop) + 1,957 from data/vpt_8xx.
- `ppo_<start>_<setting>`: PPO, starting from a BC model (`ppo_bc_6xx-304_kl05` = from
`bc_6xx-304`, target_kl 0.05) or from scratch (`ppo_scratch_...`).
- `test_...`: not a real experiment.

The runs were renamed on Nov 11 (old names like `bc_multi`, `bc_all`, `finetune_kl05` are mapped
in `runs/renamed_runs.json`).

## Results



### Main baseline: 24 worlds (seeds 1-24), episodes end on death


| policy                                                                         | mean     | median | wins (64) | 0-log games      | deaths                                  | file                                              |
| ------------------------------------------------------------------------------ | -------- | ------ | --------- | ---------------- | --------------------------------------- | ------------------------------------------------- |
| random actions | 2.5 | 1.5 | 0 | 6 | 1/24 | `runs/random/evals/20261111_155924` |
| PPO from scratch `ppo_scratch_30k` (current actions, 30k steps) | 3.0 | 1.5 | 0 | 8 | 6/24 | `evals/20261111_155924` |
| BC `bc_6xx-304`                                                                | **18.7** | 15.5   | 2         | 3                | 6/24 (all drowning)                     | `evals/20261110_114949`                           |
| BC `bc_6xx-1603` (1,603 segments, 282k examples)                               | **26.5** | 22     | 4         | 0                | 4/24                                    | `evals/20261111_105730`                           |
| BC `bc_6xx-1762_8xx-1957` (6.x treechop + 8.x, 3,719 segments, 1.13M examples) | **38.3** | 38.5   | 7         | 2 (seeds 3, 22)  | **0/24**                                | `evals/20261111_131528`                           |
| BC `bc_6xx-1762_8xx-1957_plus20` (the above + 20 more epochs, lr 1e-4)         | **42.8** | 44.5   | 4         | **0**            | 3/24 (seeds 5, 9, 18, late in the game) | `evals/20261111_142003`                           |
| BC `bc_6xx-1762_8xx-1957_plus40` (20 more again: 60 epochs in all)             | 37.5     | 36     | 7         | 1 (seed 16)      | 1/24 (seed 24)                          | `evals/20261111_145652`                           |
| BC `bc_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777` (all versions, 7,599 segments, 20 epochs) | 41.6 | 42 | 6 | **0** | 1/24 (seed 5, step 301) | `evals/20261111_211618` |
| `ppo_bc_6xx-304_kl05` checkpoint at 120k                                       | **34.3** | 30.5   | 3         | 2 (seeds 13, 22) | **1/24** (seed 5, not drowning)         | `evals/20261110_193136` + `evals/20261110_220203` |
| `ppo_bc_6xx-1762_8xx-1957_plus20_kl05` (PPO 120k from the 42.8 BC model)      | **59.8** | 64     | **19**    | **0**            | **0/24**                                | `evals/20261111_191652`                           |


The fine-tuned model beat BC on 18 of 24 worlds, lost on 4 and tied on 2: on average +15.7 logs per world
(standard error about 3). Minecraft failed partway through 4 of its games: seeds 18, 19 and 20 were
replayed (`evals/20261110_220203`, 16 -> 26, 9 -> 26, 5 -> 51), and seed 15 failed at step 999 of 1000, so
it was kept.

Videos: `evals/20261110_193136/` has all 24 seeds, but its seed 18, 19 and 20 videos are from the
games where Minecraft failed; watch the replays in `evals/20261110_220203/` for those three.
The BC videos for the same worlds are in `evals/20261110_114949/`.

`bc_6xx-1603` is the same BC recipe as `bc_6xx-304` (20 epochs, window 5 s) on all 1,603 downloaded segments
instead of 304 (5.6x the examples). Against `bc_6xx-304` on the same worlds: +7.9 logs per world
(standard error about 2.9), better on 16, worse on 6, tied on 2, and no 0-log games (BC: 3). Its
validation loss was 3.142 (`bc_6xx-304`: 3.277, but on a different held-out set). No Minecraft errors
in this eval. Not fine-tuned yet. Videos: `evals/20261111_105730/`.

`bc_6xx-1762_8xx-1957` adds the 8.x (house building) recordings to all 1,762 treechop segments, keeping only
log breaks with an axe in the inventory, at most 2 s of attacking, and before the player placed any
logs (vpt_data.py's filter). Same BC settings (20 epochs, window 5 s). Against `bc_6xx-1603`: +11.8 logs
per world (standard error about 3.8), better on 15, worse on 5, tied on 4; no deaths and no
Minecraft errors. It beats the fine-tuned `ppo_bc_6xx-304_kl05` (34.3) without any PPO. Validation loss
3.126 (held-out treechop 3.162, held-out 8.x 3.111; a new name-based split, so not comparable
with the runs above). Videos: `evals/20261111_131528/`.

`bc_6xx-1762_8xx-1957_plus20` continues that model for 20 more epochs (`bc_train.py --init`, lr 1e-4
falling to 0, epochs 21-40). Validation loss 3.126 -> 3.116 (treechop 3.162 -> 3.145), flat from
about epoch 34, while the training loss kept falling (3.113 -> 3.088): the gap between them doubled,
so more epochs would mostly memorise. Game score 38.3 -> 42.8: better on 14 worlds, worse on 7, tied
on 3, +4.5 per world with a standard error of about 3.8, so it may be noise. No 0-log games (before:
2), but 3 deaths (before: 0), all after step 800.

`bc_6xx-1762_8xx-1957_plus40` continues `_plus20` for 20 more (epochs 41-60, same lr 1e-4 -> 0).
Validation loss 3.116 -> 3.111 (treechop 3.145 -> 3.137), the train/validation gap grew to 0.039.
Game score 37.5: against `_plus20` -5.3 per world (standard error about 4.0, better on 8, worse on
14), against 20 epochs -0.8 (about 5.2). So 20, 40 and 60 epochs score the same within the noise
(38.3 / 42.8 / 37.5): after 20 epochs the small validation gains no longer show up in the game.

`bc_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777`: BC on all versions 6-10 (7,599 segments, about twice the data), 20 epochs, lr 3e-4, window 5 s, 67 min including the new 27 GB frame cache. Validation loss per version: 6.x 3.162 (same as before), 8.x 3.096 (before 3.111), 7.x 3.364, 9.x 3.127, 10.x 3.297. Game score 41.6: against the 20-epoch 6.x + 8.x model +3.3 per world (standard error about 4.1, better on 9, worse on 10), against `_plus20` -1.2 (about 3.0). So doubling the data with 7.x-10.x did not raise the game score within the noise.

`ppo_bc_6xx-1762_8xx-1957_plus20_kl05`: PPO fine-tune of `_plus20` (lr 1e-4, ent 0.001, target_kl 0.05, 120k steps, 4 envs, ~2.7 h). Game score 59.8 (median 64): 19 of 24 worlds hit the 64-log cap, no deaths, lowest seeds 11 (26) and 23 (27). Against its BC start: +17.0 per world (standard error about 4.0, better on 19, worse on 2). Training reward at the end ~55 per 2,000-step episode. So PPO still adds a lot on top of a better BC model, and the 1,000-step eval is now near its ceiling: further gains will need a harder test (more steps, or a log count beyond 64).

Random actions and PPO from scratch (30k steps, same actions and setup as now, no BC) were run on the same 24 worlds on Nov 11: 2.5 and 3.0 logs per game (from scratch minus random: +0.5, standard error about 0.9, so no better than random).

### Earlier: 8 worlds (seeds 1-8), before episodes ended on death


| policy                                                                        | mean                 | median  | wins  | file                                 |
| ----------------------------------------------------------------------------- | -------------------- | ------- | ----- | ------------------------------------ |
| random actions                                                                | 2.9 / 3.2 (two runs) | 2.5     | 0     | `20261109_221254`, `20261109_225115` |
| BC v1 `bc_6xx-303_v1-inventory-filter` (inventory filter)                     | 12.4                 | 5.5     | 0     | `20261109_221254`                    |
| BC v2 `bc_6xx-304_v2-7actions` (mine_block stats + window before a log break) | 17.9                 | 10.5    | 0     | `20261109_221942`                    |
| BC v3 `bc_6xx-304` (move/yaw/pitch actions)                                   | 21.4, re-run 26.8    | 13 / 20 | 1 / 2 | `20261109_225115`, `20261110_100622` |
| fine-tune `ppo_bc_6xx-304_kl02_no-death-end`, checkpoint at 30k               | 21.4                 | 11.5    | 2     | `20261109_234542`                    |
| fine-tune `ppo_bc_6xx-304_kl02_no-death-end`, checkpoint at 40k               | 17.6                 | 14.5    | 1     | `20261109_234542`                    |


(The short evals of 50-80 steps or a single seed, in `bc_6xx-304`'s `evals/`, were video/debug tests. A run that
showed 3,259 fake "logs" from the Minecraft-hang bug was deleted.)

### Quick check: 4 worlds (seeds 1-4), episodes end on death, `evals/20261110_162445`


| policy                                                                 | s1  | s2  | s3  | s4  | mean     | deaths |
| ---------------------------------------------------------------------- | --- | --- | --- | --- | -------- | ------ |
| random                                                                 | 0   | 2   | 11  | 7   | 5.0      | 0/4    |
| BC `bc_6xx-304`                                                        | 4   | 17  | 16  | 64  | 25.2     | 0/4    |
| `ppo_bc_6xx-304_kl05` checkpoint at 50k                                | 16  | 34  | 64  | 64  | 44.5     | 0/4    |
| `ppo_bc_6xx-304_kl05` final (~120k)                                    | 31  | 39  | 60  | 61  | **47.8** | 0/4    |
| `ppo_bc_6xx-304_kl05` checkpoint at 120k (2nd eval, `20261110_192334`) | 38  | 47  | 64  | 64  | 53.2     | 0/4    |
| `ppo_bc_6xx-304_kl05` final (~163k, `20261110_192334`)                 | 26  | 40  | 64  | 64  | 48.5     | 0/4    |


BC on these seeds across all 4 evaluations so far: s1 9/0/1/4, s2 26/21/31/17, s3 64/64/32/16,
s4 32/64/64/64 (about 32 on average). Both fine-tuned checkpoints beat every BC run on seeds 1 and 2,
so this is the first real sign that fine-tuning helps. Only 4 worlds, though: confirm on 24.

## PPO fine-tuning runs (from `bc_6xx-304`)

All runs: lr 1e-4, ent_coef 0.001, n_steps 1024 x 4 envs = 4,096 steps per update, 10 epochs,
batch 64, 5 value warm-up updates (only the value head trains).


| run                   | target_kl | steps                                           | result                                                                                                                                                                                                                           |
| --------------------- | --------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ppo_bc_6xx-304_kl02` | 0.02      | ~94k (stopped)                                  | Every update early-stopped at epoch 0 (KL ~0.04 > 0.03 limit). ep_rew_mean flat at ~21-24; explained_variance rose to ~0.6-0.75. No forgetting, but hardly any learning.                                                         |
| `ppo_bc_6xx-304_kl05` | 0.05      | ~163k (120k with 4 envs, then +40k with 3 envs) | Most updates complete several epochs. At ~100k: ep_rew_mean rose from ~20 to ~34, explained_variance ~0.6-0.8, approx_kl ~0.03-0.046, clip_fraction ~0.3-0.4 (high), entropy 4.1 -> 4.65 (max is 5.69). Needs the 24-world eval. |


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



### Fine-tuned (120k) failure modes, from the 24-world videos

- **Drowning is gone**: 0 drownings in 24 games (BC: 6).
- **Digging into the ground** (seed 22, 0 logs): attack is always held, so looking down digs. It dug
itself into a dark pit (19 dirt in the inventory) and stayed there from about step 300 to the end.
- **Caves / darkness** (seed 5, the only death): with 8 logs it wandered into a dark stone area,
lost health and died at step 783 (looks like a mob; the death message is too small to read).
- **Bad terrain** (seed 13, 0 logs): spawned by water among hills, saw birch trees but never got to a
trunk, spent the time jumping around and looking at leaves or the sky.



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

- Confirm `ppo_bc_6xx-304_kl05` on the 24 worlds (the 4-world check: 47.8 vs BC 25.2).
- From 120k to 163k, ep_rew_mean stayed flat at ~38 and the 4-world score didn't change (120k: 53.2, 163k: 48.5):
more PPO at these settings has stopped paying off. Next, find what still fails on 24 worlds.
- If fine-tuning loses BC skills: add a KL penalty toward the BC policy.
- If the reward collapses: lower target_kl or the learning rate (clip_fraction is already high).

