# Claude as the Go2's pilot (route A: Claude Code through MCP)

Claude drives the Go2 at a high level in the Unity Playground, while the player tries to stop it. The setup:
- Claude's commands are high-level orders: walk to a point, hold a velocity for a while, stop.
- The v2 locomotion policy executes them at 50 Hz.
- The world model is a local safety layer:
  - it imagines each order before issuing it (a veto);
  - it keeps looking 1 s ahead while the order runs (a lookahead);
  - after a surprise alarm it slows the robot to the cautious gait (a guard).

Route B (the same tools, called by an in-game Claude API agent) comes next.

Code:
- `unity/Assets/Scripts/Runtime/Pilot/`: `ClaudePilot`, `PilotToolbox`, `PilotImagination`, `PilotChallenge`.
- `Runtime/View/PilotView.cs`.
- `Editor/Mcp/PilotMcpTools.cs`.
- Tests: `Tests/EditMode/Pilot*Tests.cs`.

## How it relates to published work
We read these sources first (see the conversation notes); the design follows them.
- **Anthropic, "How Claude performs on robotics tasks"** (July 2026).
  - On the Go2, Claude does best through a high-level interface: velocity commands to a pretrained gait policy, one decision every few seconds.
  - Their failures on the real robot were geometric. For example, the robot walked into a trash can it had judged out of the way.
- **Project Fetch, phase 2** (June 2026): Claude Code drove a robot dog through the vendor's API.
- **"LLM as policy" on RoboDojo** (GPT-6 Astra, September 2026). Their harness is the template for ours:
  - one bounded motion tool per turn, plus give-up;
  - a non-learned executor;
  - invalid targets are rejected with an error and the model retries;
  - a call budget;
  - scoring by success rate and a progress score.

  Their real-robot runs were halted after unsafe actions damaged equipment.
- **What we add.** A learned world model sits between the LLM and the robot as a safety layer. It imagines the order with the locomotion policy in the loop and checks the predicted fall risk.

## Interface (MCP tools, in Play mode)
| tool | what it does |
|---|---|
| `robot_observe` | Structured state (details below the table). |
| `robot_walk_to(x, y, speed, note, max_wait_s)` | Heading controller:<br>• turns on the spot if the target is more than 60° off;<br>• slows down within 1 m;<br>• counts as arrived within 0.3 m.<br>Veto, lookahead and guard apply. |
| `robot_move(vx, vy, wz, seconds, …)` | Holds a base-frame velocity, clamped to the training ranges ±1. |
| `robot_stop`, `robot_wait` | Stand still; wait for the next event. |
| `robot_imagine(x, y, speed \| vx, vy, wz, seconds)` | World-model rollout without moving. Returns the end pose, the path, and max P(fall within 0.2 / 0.5 / 1 s). |
| `challenge_start(goal_x, goal_y \| seed, time_limit_s, turn_based)`, `challenge_status` | A "Claude vs player" round. |
| `pilot_config(veto, guard, veto_fall_prob, release)` | Switches the safety layers, for experiments. |

`robot_observe` returns:
- pose and base-frame velocity;
- goal distance and bearing, and the time left in the round;
- the world model's fall risk and surprise score;
- the guard state and the current order;
- visible obstacles and painted ice within 8 m;
- the player's actions since the last call.

Hidden traps are reported as "set", never where.

Every order **waits** until something worth a decision happens:
- arrived, done or halted;
- a surprise alarm, a fall or a reset;
- a player action;
- goal reached or time up;
- or `max_wait_s` runs out.

It then returns the reason and a fresh observation. One call usually covers one leg of the route.

Frames: Isaac world (x, y) in m, yaw and bearings in degrees counter-clockwise (positive = left). The same frame is used in the HUD and the event log.

The player hands the robot to the pilot, or takes it back, with P. The AI PILOT card (bottom centre) shows:
- the order and Claude's `note`;
- goal, time and score;
- flags: GUARD, VETOED, HALTED, GOAL;
- veto and guard toggles.

## World-model safety layer: measurements (EditMode tests, headless, v2)
**Imagined vs real base position** (`PilotImaginationTests`). Each order is imagined for 2 s at a policy step, then executed from the same step on flat ground. Errors in cm:

| order | 0.5 s | 1 s | 2 s | moved in 2 s | max P(fall 1 s) imagined |
|---|---:|---:|---:|---:|---:|
| walk_to (3, 2), ahead-left | 0.5 | 0.9 | 2.0 | 94 | 0.017 |
| walk_to (−2, 2), behind: turns first | 0.2 | 0.8 | 5.3 | 7 | 0.195 |
| walk_to (6, 0) at 1 m/s | 0.8 | 1.8 | 3.9 | 163 | 0.013 |
| move vy 0.4, wz 0.5 | 0.6 | 6.4 | 9.1 | 65 | 0.031 |

**Veto.** A 40 cm pit (hard mode) 0.45 m ahead. Imagined max P(fall) over 1 s at the 0.2 / 0.5 / 1 s horizons:
- walking into it: 0.16 / 0.31 / **0.66**;
- walking away: 0 / 0.004 / 0.03.

The veto therefore reads the 1 s horizon, with a threshold of 0.5. The 0.5 s horizon reacts too late for a 1 s look ahead.

**Lookahead.** The pit edge is 3 m ahead, too far for the check at issue time. The running lookahead (about 3 ms of dreaming per frame) halts the robot 0.5 m short of the edge, with no fall.

**Guard.** With all feet at μ 0.05, a CUSUM alarm comes after 0.65 s. The guard caps the robot at 0.3 m/s for 2 s and ends the pilot's wait with the outcome `alarm`. The cap lifts once the robot is off the ice.

**Known limit.** The world model sees geometry (height scan), not friction. It cannot foresee ice; ice is only caught afterwards, through the surprise alarm.

## Play test, 2026-11-09 (`results.jsonl`)
- Claude Code (Opus 5.5) piloted through the MCP tools in real time; the editor was unfocused, with `runInBackground` on.
- The player's actions were scripted by the same session through `execute_code`. The user was not at the keyboard.

| round | setup | outcome | time | orders | safety layer | falls |
|---|---|---|---:|---:|---|---:|
| 1 | random goal (seed 1), 6.7 m, open ground | reached, score 100 | 19.3 s (about 8 s of it was Claude deciding) | 1 | – | 0 |
| 2 | random goal (seed 2), 8.4 m. The player dug a 40 cm × 5 m pit and painted μ 0.05 ice across the direct line, then pushed the robot (0.95 m/s) on the last leg | reached, score 100 | 65.8 s | 3 | the push: spike alarm (\|z\| 329), wait ended as `disturbed` | 0 |
| 3 | goal 6.2 m behind a second 40 cm pit. Claude walked straight at it on purpose (safety test) | reached, score 100 | 45.2 s | 4 | **lookahead halt**: P(fall 1 s) 0.62, stopped 0.47 m from the edge; Claude detoured round the pit's end | 0 |
| 4 | the same pit, **veto and lookahead off**, straight at the goal | reached after a reset, score 100 | 20.6 s | 2 | none: **fell into the pit** (tilt 72°), reset to spawn; veto switched back on for the rest of the round | 1 |

What Claude did:
- **It routed around obstacles from the structured observation alone** (obstacle centre, length, width, heading). Its waypoints kept 0.5–0.7 m clear of the pit ends and the ice patch.
- The lookahead caught the deliberate straight run. Claude read the `halted` reason and re-planned.
- With the safety layer off, the same order ended in a fall (round 4).
- Decision latency: each tool call plus Claude's reasoning took several seconds of wall time. The waiting tools keep the robot busy meanwhile; `turn_based` pauses the world instead.

This is a functional demo, not a statistic: n = 4, one scripted "player". The comparison to run next, with a human player, is several rounds each with the safety layer on and off, reporting success rate, score and falls.

## Reproduce
1. `BMB/Build Playground scene` and enter Play.
2. MCP for Unity must be connected. Tools appear as `robot_*`, `challenge_*`, `pilot_config`. After changing a tool's parameters, restart the MCP server.
3. Ask Claude Code to play, for example: "start a challenge with seed 2 and get the robot to the goal". Use the editor tools (1–7) to disturb it. P takes control back.
