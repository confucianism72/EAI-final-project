---
description: How to run ManiSkill with GPU and network access
---

# Running ManiSkill Commands

## Proxy Setup (Required for downloads)
// turbo
1. Set proxy environment variables:
```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890
```

## Common Commands

// turbo
2. Test environment (single env, no DR):
```bash
python -m scripts.test_env --task lift --no_dr
```

// turbo
3. Test with domain randomization:
```bash
python -m scripts.test_env --task lift
```

// turbo
4. Multi-env GPU test:
```bash
python -m scripts.test_env --task lift --num_envs 4
```

// turbo
5. GUI mode:
```bash
python -m scripts.test_env --task lift --gui
```

## Task Options
- `--task lift` - Lift red cube (≥5cm)
- `--task stack` - Stack red on green
- `--task sort` - Sort cubes to grids
