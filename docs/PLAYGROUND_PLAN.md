# Phase 5 Playground & Regression Plan

## Work Items

### A. Local CPU Smoke Tests
Verify GPU buffer changes don't break numpy path after `ecc884c9` + `a7dcd8ee` commits.

- Run `ppo_cartpole` (PPO, numpy replay, classic control regression)
- Run `sac_cartpole` (SAC, numpy replay, discrete)
- Run `crossq_cartpole` (CrossQ, BRN, regression check)
- Expected: all complete without error, scores within normal range

Commands:
```bash
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
uv run slm-lab slm_lab/spec/benchmark/sac/sac_cartpole.json sac_cartpole train
uv run slm-lab slm_lab/spec/benchmark_arc/crossq/crossq_cartpole.yaml crossq_cartpole train
```
Status: ⬜ TODO

---

### B. Phase 1-4 Regression Runs (dstack GPU)
Confirm fps same or better, scores within variance vs HF benchmark data.
One representative per phase, plus SAC MuJoCo for GPU buffer regression.

| Phase | Env | Spec | Target fps | Target score |
|-------|-----|------|-----------|-------------|
| 1 | CartPole-v1 | ppo_cartpole | ~3000fps | ~490 |
| 2 | LunarLander-v3 | ppo_lunar | ~1200fps | ~130 |
| 3 | HalfCheetah-v5 | sac_halfcheetah_arc | ~200fps | ~9815 |
| 3 | HalfCheetah-v5 | crossq_halfcheetah | ~700fps | ~8085 |
| 4 | ALE/Pong-v5 | ppo_atari_arc (-s env=ALE/Pong-v5) | ~700fps | ~19 |

Launch after C.5 dstack unification is done (so playground flag works correctly).
Status: ⬜ Blocked by C.5

---

### C.1. BENCHMARKS.md Phase 5 Polish

Issues to fix:
1. "Install: `uv sync --group playground`." buried in Environment sentence — move to separate prominent line
2. Algorithm Specs line needs to include PPO (not only SAC)
3. Running command uses wrong spec path and `--playground` flag (remove after C.5)
4. Table has per-env Spec column — replace with single shared spec + `-s env=X` pattern (after C.3)
5. Add Phase 5 row to Progress table
6. Add `--playground` removal note after dstack unification

Status: ⬜ TODO (depends on C.3 for table format, C.5 for command format)

---

### C.2. Spec Refactor: JSON → YAML with TorchArc

Current: 4 JSON files in `slm_lab/spec/benchmark/playground/`
Target: YAML files in `slm_lab/spec/benchmark_arc/playground/` with TorchArc architecture

Files to create:
- `ppo_playground_arc.yaml` — PPO shared template with `${env}` and `${max_frame}`
- `sac_playground_arc.yaml` — SAC shared template (DM Control + Locomotion + Manipulation)

YAML pattern to follow: `sac_mujoco_arc.yaml` (anchors, TorchArcNet, `${env}` substitution)

Files to delete: all 4 JSON playground specs

CLAUDE.md updates needed:
- Line 131: "JSON specs" → "YAML specs"
- Line 164: "JSON configuration" → "YAML configuration"
- Line 172: "All experiments defined in JSON" → "All experiments defined in YAML"
- Lines 186, 189, 199, 200: Update .json → .yaml paths in quick test commands

Status: ⬜ TODO

---

### C.3. Spec Reorganization: Atari-Style Single Spec

Current: separate spec per env (58 specs!) or per-group (4 files)
Target: one shared spec per algo, `-s env=playground/CartpoleBalance` substitution

Pattern from Phase 4:
```bash
slm-lab run-remote --gpu -s env=ALE/Pong-v5 slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_arc train -n pong
```

Playground equivalent (after C.5, no --playground needed):
```bash
source .env && slm-lab run-remote --gpu -s env=playground/CartpoleBalance slm_lab/spec/benchmark_arc/ppo/ppo_playground_arc.yaml ppo_playground_arc train -n pg-cartpole
```

Status: ⬜ TODO (part of C.2)

---

### C.4. Phase 5 Benchmarks (dstack runs)

**Wave 1 — PPO simple DM Control (verify algo works on playground)**
Priority envs (easy, clear solve targets from DM Control paper):
- CartpoleBalance (target ~1000)
- PendulumSwingup (target ~700-800)
- ReacherEasy (target ~950)
- PointMassEasy (target ~970)

**Wave 2 — SAC medium DM Control**
- CheetahRun (target ~800)
- WalkerWalk (target ~950)
- HopperHop (target ~800)

**Wave 3 — SAC hard + Locomotion robots**
- HumanoidWalk (target ~800)
- WalkerRun (target ~700)
- Locomotion: H1Walking, SpotJoystickFlatTerrain

DM Control scores from original DrQ/DreamerV2 papers (normalized 0-1000 scale):
easy envs typically >900 with any decent RL algorithm at 1M steps.

Status: ⬜ Blocked by C.2+C.3 (need YAML specs first)

---

### C.5. dstack YAML Unification

Changes:
1. Remove `.dstack/run-gpu-playground.yml`
2. Generalize `.dstack/run-gpu-train.yml`:
   - Add `PLAYGROUND` to env section
   - In commands: `if [ -n "$PLAYGROUND" ]; then uv sync --group playground; fi` (or conditional install)
   - Add `XLA_PYTHON_CLIENT_PREALLOCATE: "false"` to env section
   - Fix `max_duration: 8h → 6h`
3. Update `slm_lab/cli/remote.py`:
   - Remove `--playground` config file selection
   - Pass `PLAYGROUND=true` env var when `playground=True`
4. Update README/docs to remove `--playground` flag documentation

Status: ⬜ TODO

---

## Team Assignments

| Agent | Tasks |
|-------|-------|
| infra-agent | C.5 (dstack YAML unification) |
| specs-agent | C.2 + C.3 (YAML spec conversion + reorganization) |
| docs-agent | C.1 (BENCHMARKS.md) + CLAUDE.md JSON→YAML updates |
| test-agent | A (local CPU smoke tests) |

B (Phase 1-4 GPU regression) and C.4 (Phase 5 benchmarks): supervised by lead after C.2+C.5 complete.

---

## Dependencies

```
C.5 ──────────────────────────────────────────► B (launch regressions)
C.2+C.3 ──────────────────────────────────────► C.4 (launch benchmarks)
C.2+C.3+C.5 ──────────────────────────────────► C.1 (update commands in docs)
A (independent, run immediately)
```
