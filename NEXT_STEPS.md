# Next steps

Ideas for what to try next, based on the results in [EXPERIMENTS.md](EXPERIMENTS.md).
Roughly in the order I'd try them.

## Where we are (Nov 10, 2026)

- Best model: `runs/model_20261110_133518_finetune_kl05/treechop_ckpt_120000_steps.zip`
  (BC `bc_multi` + PPO fine-tuning with target_kl 0.05).
- On 24 worlds: **34.3 logs per game** (BC: 18.7), 1 death (BC: 6, all drowning), better than BC on
  18 of 24 worlds.
- PPO has plateaued: from 120k to 163k steps the training reward stayed at about 38 and the eval
  score didn't change.
- What still fails (from the videos):
  1. **Digging into the ground**: attack is always held, so looking down digs a pit it can't get out of.
  2. **Caves and darkness**: wanders into dark stone areas, where a mob killed it.
  3. **Searching**: once nearby trees are gone, or in hilly/watery terrain, it doesn't find new ones.

## 1. Quick fixes to the tooling (small, do first)

- **Retry failed games in `evaluate.py`.** Minecraft failed in 4 of 24 games in the last eval, and
  3 had to be replayed by hand. Automatically replaying a game that ends with `env_error` would make
  every eval trustworthy without manual work.

## 2. Let the agent choose whether to attack (fixes digging pits)

The biggest remaining failure is self-inflicted: attack is always on, so looking down digs.

- Add a 4th action part, `attack: [off, on]`, to make the action space `[3, 11, 9, 2]`.
- The VPT data already records whether the human was attacking (`attack` in each `.npz`), so BC can
  learn it with no new downloads: update `labels()` and re-run `bc_train.py`, then fine-tune again.
- **Risk:** the agent might learn to attack less often and chop more slowly. Compare logs per game
  and look at the videos for "stopped swinging while facing a log".
- **Cheaper alternative to try first:** keep attack always on, but end the episode (or give a small
  penalty) when the agent is stuck: no logs and almost no camera change for N steps, or more than
  some amount of dirt/stone in the inventory. This doesn't fix the cause, but it gives PPO a signal
  that pits are bad.

## 3. Help it find trees (fixes searching)

- **Look further ahead:** raise `gamma` from 0.99 to 0.995. A log 30 s away would then count about
  half, instead of almost nothing. Trees out of sight would matter more. Needs a value warm-up again,
  because the value scale changes.
- **Give it a short memory:** stack the last 2-4 frames as the observation. Right now it sees one
  frame, so it can't tell which way it was already turning or whether it's going in circles. This
  changes the network input, so it means re-running BC (the VPT frames can be stacked the same way).
- **See more:** 64x64 is low resolution, and distant trees are only a few pixels. Try 96x96 or
  128x72. This also means re-running BC.
- **Exploration bonus** (only if the above don't help): a small reward for reaching new places.
  It needs the player's position (a new MineRL observation, so test for hangs), and it can be gamed:
  running around instead of chopping. Keep it small compared with +1 per log.

## 4. A better starting point (improve BC itself)

Every improvement to BC carries over to fine-tuning (BC 18.7 -> fine-tuned 34.3).

- **Use more of the VPT data:** we use 304 of the 4,049 treechop segments. Downloading more (e.g.
  1,000+) is cheap compared with Minecraft time.
- **Try a longer window** before log breaks (`--window 10` instead of 5) to copy more of how humans
  search for the next tree, which is exactly the weak spot.

## 5. If PPO stays stuck at its plateau

- **Lower the learning rate as training goes** (e.g. 1e-4 -> 3e-5), or lower target_kl to 0.03: the
  clip_fraction was 0.3-0.4, which is high, so updates may now be too big to make fine progress.
- **Bigger batches:** more envs or a larger `n_steps` give less noisy updates. More envs need more
  memory (about 3.3 GB per Minecraft instance).
- **Switch to an off-policy method that also uses the human data** (SQIL / DQfD-style: demos in the
  replay buffer). This reuses each Minecraft step many times instead of once, and is what won the
  MineRL competitions. Bigger change: a new training script and flattening the actions.

## 6. Measuring

- **Evaluate with 2,000-step games** too: training episodes are up to 2,000 steps, and searching
  matters more over longer games.
- Always compare on the same 24 worlds, and replay Minecraft failures.
- Still missing: a random-action baseline on the 24 worlds.
