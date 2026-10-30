# Speed Lab results (measured only)

| Experiment | Configuration | Time | Speedup | VRAM | Tests | Decision |
|---|---|---:|---:|---:|---|---|
| Reference | Base Soup (full ×2, 0c58998 session) | 87.39 / 80.31 s | 1.0x | 14345 MiB | — | Reference |
| Current | 0c58998 default | 70.99 / 71.04 s | 1.181x | 3765 MiB | 323/0/0, Soup 79, 10/10 | Best known |
| wave1 A | 6e62dbf default (session base 76.96 s) | 68.60 s | 1.122x | 3765 MiB | unit 331/0/0 | reference for the session |
| wave1 B | SOUP_EXP_PRECOMPILE=1 | 68.10 s (Δ −0.46, CI [−1.20, +0.10]) | 1.130x | 3769 MiB | bitwise cache-hit PASS | loop −5.6 s but moved to startup; not significant alone |
| wave1 C | SOUP_EXP_LAUNCH_TRIM=1 | 68.09 s (Δ −0.46, CI [−1.25, −0.06]) | 1.130x | 3765 MiB | bitwise PASS | small real gain |
| wave1 D | B + C | 67.94 s (Δ −0.75, CI [−1.36, −0.31]) | 1.133x | 3769 MiB | bitwise PASS | best, ~1%: candidate default |
