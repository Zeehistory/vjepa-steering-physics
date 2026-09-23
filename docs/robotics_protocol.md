# Robotics experiment (paddle strike): protocol

This document describes the procedure step by step: what is simulated, what the model sees, what is
trained, and how every reported number is measured.

Code: `experiments/threads/paddle-robotics/` (scripts), `src/data/{paddle_strike,franka_strike,franka_dynamic_strike}.py`
(scenes), `src/control/{strike_inverse,action_predictor}.py` (models).
Outputs: `outputs/paddle_strike/`.

---

## 1. The question

> Given a video of a ball coming towards a robot, and a desired outgoing ball speed, can we read
> off V-JEPA's latent space the action the robot must take to produce that speed — and does
> the ball actually leave at that speed when we execute it?

The robot perceives only pixels, through V-JEPA 2-L. Success is **measured in the simulator in
m/s**. It is not a latent-space or regression score.

## 2. The task (the scene)

A MuJoCo simulation of a **head-on return**, seen from above at a 42° camera elevation:

1. A ball slides along a frictionless table in the +x direction at speed `v_in`.
2. A striker (paddle) waits at a **fixed** position, then accelerates to a commanded speed `v_p`
   in the −x direction.
3. They collide and the ball bounces back (−x) at speed `v_out`.

**The action** is one number, the striker speed `v_p` (m/s).
**The controlled outcome** is the ball's outgoing speed `|v_out|`.

The physics obeys a linear law that we measure (we don't assume it):
`v_out = 1.8995·v_p − 0.8986·v_in`. It is used only as a reference ceiling. The policy never
sees it.

Why head-on: V-JEPA sees displacement per frame, so a paddle chasing the ball from behind runs out
of frame before the clip ends. A head-on hit keeps everything in view.

### Three embodiments (keep them apart)

| name | what hits the ball | used for |
|---|---|---|
| `paddle` | an abstract paddle on a slider | main dataset |
| `franka` | the same paddle, **held by a Franka Panda arm** that is posed by inverse kinematics but is *kinematic* (it adds no inertia) | appearance transfer: identical physics (agrees to 1e-12 m/s), different pixels |
| `franka_dynamic` | a **genuinely actuated** Panda: link inertia, torque servos, a sprung tool mount | **execution only**: tests whether a real arm can carry out the chosen action |

`franka_dynamic` is never shown to the encoder. Its swing starts at frame 4, inside the context
window, so showing it would leak the action. The policy perceives the kinematic `franka` rendering,
and the action it picks is then executed on the actuated arm.

## 3. The data

Each **scene** fixes an incoming speed `v_in ∈ [0.85, 1.15]` m/s and 8 commanded **ratios**
`|v_out|/v_in ∈ [0.5, 3.0]`. Each (scene, ratio) pair is one episode, and every episode yields two
clips:

| clip | contents | count |
|---|---|---|
| **pre** (context) | the ball approaching, striker at rest. **Bit-identical across all 8 actions in a scene**, so the context cannot reveal the action. | 1 per scene |
| **post** (outcome) | the ball leaving at `v_out`. The window opens when the ball crosses a fixed x, so all 8 post clips start with the ball at the same pixel (spread 0.56 px) and **differ only in speed** | 8 per scene |

Clip format: **16 frames, 256×256, 60 fps** (this is not the 4 fps of the ball scenes). The collision
itself always falls in an unrendered gap between the two windows.

Splits are scene-disjoint and set by seed:

| split | seed | scenes | post clips | pre clips |
|---|---|---|---|---|
| train | 0 | 500 | 4000 | 500 |
| test | 2 | 100 | 800 | 100 |

Labels (`v_in`, `v_p`, `v_out`) are **measured from the simulator**, never copied from the command.
Every run regenerates the physics deterministically from (seed, index) and checks it against the
cached clip's image-plane velocity. They agree to 1.9e-9.

Before any GPU work, `experiments/pipeline/01_data/validate_paddle_strike.py` certifies the scene on
CPU (about 14 s). It sweeps actions, fits the law, and checks that 240 held-out commands land within
±5%. Result: 100%, worst case 0.25%, on both `paddle` and `franka`.

## 4. The pipeline, step by step

### Step 1: Encode (GPU)
`02_encode/extract_paddle_strike.sh` runs every pre and post clip through **frozen V-JEPA 2-L** and
saves the token grid at **layers 6, 12, 18, 23**. Each layer gives 8 time × 16×16 space tokens ×
1024 dims.

### Step 2: Reduce to a descriptor (CPU)
For each clip and layer, the 16×16 spatial grid is average-pooled to 4×4. The 8 time steps are kept
and the 4 layers are concatenated, giving 524,288 numbers per clip. A **PCA with k = 64 components
is fit on train post clips only** and applied to every clip (pre and post, train and test). All
later steps work on these 64-d vectors: `h_pre` for context, `z_post` for outcome.

### Step 3: Fit the steering map `W_tgt` ("imagine the outcome")
A ridge regression that learns what the outcome latent *would* look like for a given speed:

```
z_post ≈ W_tgt · [h_pre, v_out]
```

At test time we hand it `[h_pre, v*]` with the **desired** speed `v*`, and it returns a synthetic
target latent `z_target`. This is the "steered belief": an outcome that has not happened. It is fit
on train scenes only.

### Step 4: Train the action predictor ("which action gets me there")
`03_train/train_action_predictor.py`, with the model in `src/control/action_predictor.py`:

```
π : [z_target, h_pre, z_target − h_pre]  →  v_p
```

- Families: ridge, and an MLP (a linear skip plus a zero-initialised deep correction, Huber
  loss, 5-member ensemble, per-column standardisation). It runs on CPU in minutes.
- **Training targets** (`target_source`): `real` = the real post latent, `synth` =
  `W_tgt([h_pre, true v_out])` (what test time actually provides), `both` = both stacked. The MLP
  trained on `real` alone fails at deployment (7.4% error vs 0.09% with `both`), so `both` is used.
- The ridge penalty for `W_tgt` is selected by **end-to-end executed error** on 10% of the train
  scenes held out as validation (50 scenes). Latent reconstruction error is not used for selection.

### Step 5: Execute and measure (the actual test)
For each of the 100 test scenes and each of 8 commanded ratios
(0.75, 1.0, 1.35, 1.7, 2.0, 2.3, 2.6, 2.9 × `v_in`), so 800 commands in total:

1. Encode the scene's pre clip → `h_pre`.
2. `z_target = W_tgt([h_pre, v*])`.
3. `v_p = π(z_target, h_pre)`.
4. **Run the strike in MuJoCo with that `v_p`** and read the ball's actual `v_out`.
5. Error = `| |v_out| − v* | / v*`.

Reported: **pass@±5%** (the pre-registered tolerance; e.g. a 2 m/s command must land in 1.9–2.1 m/s),
pass@±2%, and median error.

### Step 6: Calibration (actuated arm only)
The predictor's output constants are re-identified on the rig. On 12 held-out train scenes we
regress achieved speed on commanded speed (two numbers, slope and intercept) and invert that
line. A physical robot can do the same without knowing the physics law.

## 5. Comparison arms (all on identical test scenes and identical execution)

| arm | what it does | role |
|---|---|---|
| `pred` / `pred_cal` | the predictor, without and with calibration | the method |
| `argmin` / `argmin_cal` | the earlier approach: a learned forward model `ẑ(a) = P(h_pre, a)`, then pick `a* = argmin‖ẑ(a) − z_target‖` over a 400-point action grid. It gets **its own** ridge penalty, selected the same way | baseline |
| `analytic` | the certified physics inverse | ceiling |
| `wrong_ratio` | same scene, but given the target for a *different* ratio | shows the action comes from the steered target |
| `shuffled` | given another scene's target | weak control (scenes differ only slightly in `v_in`) |
| `constant` | always commands the envelope midpoint | "no information" floor |

Significance: paired bootstrap resampled **by scene** (10,000 resamples), because the 8 ratios of
a scene share one episode.

## 6. The four settings run

| setting | perceive on | execute on | command |
|---|---|---|---|
| paddle | `paddle` | `paddle` | `EMBODIMENT=paddle sbatch …/03_train/train_action_predictor.sh` |
| Franka | `franka` | `franka` | `EMBODIMENT=franka sbatch …` |
| appearance transfer | fit on `paddle`, test on `franka` | `franka` | `TRANSFER=1 sbatch …` |
| actuated arm | `franka` | **`franka_dynamic`** (25 test scenes, 2.9 s/rollout) | `EMBODIMENT=franka EXEC=franka_dynamic sbatch …` |

Headline: 100% pass@±5% in all four settings, median error
0.10–0.13% (transfer: 2.6%), against 72–84% for the argmin baseline. `wrong_ratio` scores 0%.

## 7. What the other scripts in the folder are

| folder | script | purpose |
|---|---|---|
| `01_data` | `fetch_menagerie.sh`, `hardware_specs.py` | fetch the Panda model; hardware/regime measurements |
| `02_encode` | `extract_paddle_strike.sh` | Step 1 |
| `03_train` | `train_action_predictor.{py,sh}` | Steps 2–6 and the whole comparison |
| `05_steering` | `vector_steer.*`, `demo_steering_*` | 2-D (direction + speed) steering variant, steering demos |
| `07_eval` | `ablate_action_predictor.*` | 29 ablations (model, inputs, layers, k, train size, noise, extrapolation) |
| | `dynamic_arm_certify.py`, `franka_feasibility.py` | can the actuated Panda physically do it (joint-velocity/torque limits) |
| | `friction_robust.py`, `regime_certify.py`, `recertify_paddle.sh` | robustness of the physics law |
| `08_figures` | `demo_paddle_strike.*`, `filmstrip.sh`, … | filmstrips and videos |
| `11_run` | `close_action_loop*.{py,sh}` | the earlier argmin loop (+ affine calibration) |
| | `validated_system.*`, `onrig_calibrate.py` | law transfer under friction/compliance; on-rig re-identification |
| | `franka_hardware_export.py` | exports a 1 kHz joint trajectory in libfranka format |

## 8. What this experiment does *not* show

- The action is **one scalar** (striker speed), executed by the arm through IK and servos. It is
  not a joint-space policy.
- `z_target` is linear in `v*`, so the predictor is essentially recovering the physics inverse
  *through* the latent. The claim is that **V-JEPA's latent space encodes incoming and outgoing
  speed linearly and precisely enough to act on**. It is not a claim that the model learned
  physics beyond that.
- Everything is simulated. The Franka result says what a real Panda should do (and its limits:
  joint velocity binds, torque does not), but the constants must be re-measured on a real rig.
