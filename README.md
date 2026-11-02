# Strata vs llama.cpp on 2x Tesla P100: same model, same box, standard benchmarks

Qwen3.8-Flash-Next, two engines, one machine, the same questions. Strata is about 3x faster. Accuracy is almost the same: llama.cpp got 3 more questions right, out of 180.

| | llama.cpp (our P100 build) | Strata v0.2.2 + qr0n P100 patches |
|---|---|---|
| GSM8K, test items 0-99 | **99 / 100** | 97 / 100 |
| HumanEval+, tasks 0-79 (plus tests) | **78 / 80** | 77 / 80 |
| median decode during the run | 25.3 / 26.7 tok/s | **74.9 / 85.4 tok/s** |

Measured 2026-11-02, single run, greedy (temperature 0), thinking off.

**Where they differ.** Three questions were right on llama.cpp and wrong on Strata (GSM8K 12 and 45, HumanEval+ 10); none went the other way. A 3-0 split is not statistically significant (exact two-sided p ≈ 0.25), so treat it as a hint, not proof.

**Why the misses happened.** Strata's two GSM8K misses hit the 1,024-token limit while it was still double-checking, so it never wrote a final answer. llama.cpp's GSM8K 12 also hit the limit, but the last number it wrote happened to be the right one, so it scored as correct.

**Both wrong:** GSM8K 85 (both answered 48; the key says 44).

## Setup
- **Box:** Dell T5810, Xeon E5-2690 v4 (AVX2), 78 GB DDR4, 2x Tesla P100 16 GB on PCIe 3.0 x16, hard disks only. No SSD.
- **Model:** ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (2 shards, 75.8 GB). The same files went into both engines.
- **llama.cpp:**
  - Our own P100 build of llama.cpp (not linked here; plain upstream runs the same flags, slower). Flags: `--fit on --fit-target 1024 -sm layer -fa on -ctk f16 -ctv f16 --moe-cache-mib 12288 -b 1024 -ub 1024 -t 14`.
  - Env: `GGML_CUDA_P100=1 GGML_CUDA_P100_V2=1 GGML_CUDA_REGISTER_HOST=1`. No speculative decoding.
- **Strata:**
  - Strata v0.2.2 (011c9e60) with qr0n's 12 P100 patches (https://github.com/qr0n/p100-strata, commit 8b1b0f5). Built for sm_60 with CUDA 12.9.
  - Config in `configs/strata-cfg.json`: qr0n's "peer" layout, `--ple-io ram`, int8 KV, MTP spec 4 / min-p 0.5, `STRATA_ROUTE_TAIL_SKIP=0`, plus qr0n's fp16 switches.
- **Prompts:** our own client (`scripts/gen.py`) over the OpenAI-compatible chat API; prompts are in the script.
  - GSM8K: zero-shot chat, "The answer is <number>" on the last line. Max 1,024 tokens.
  - HumanEval+: "complete the function in one python block". Max 1,536 tokens.
- **Scoring:** `scripts/score.py`.
  - GSM8K: the number after "The answer is" (otherwise the last number) is compared with the gold answer.
  - HumanEval+: the code block plus the plus tests from evalplus v0.1.10 (OriginFmt) run in a fresh Python, 30 s timeout.
  - This is not lm-evaluation-harness and not the official evalplus runner. The data sets are the official ones.

## Reproduce
Every step, the build, the configs and the traps we hit: **[REPRODUCE.md](REPRODUCE.md)**. Short version: `scripts/get-data.sh`, then `scripts/gen.py PORT MODEL OUTDIR` against each server, then `scripts/score.py DATA_DIR a.jsonl b.jsonl`.
Raw answers of both engines are in `results/`.

## Possible reasons for the differences
These are Strata's own statements; we did not test which one matters.
- Strata's log: experts cached on the GPU "round differently from the CPU, so a reply can differ slightly".
- Strata's log: the prefill CPU share is on by default; "answers can differ slightly".
- Strata issue #2685 also reports run-to-run nondeterminism with the same seed.

## Outside these benchmarks
- **A non-standard task:** on our own payment-reminder task (a three-month total with late fees), Strata gave a wrong total in 4 of 4 runs. llama.cpp got it right in its one run.
- **Speed on our box:** 16 fresh prompts gave Strata a median of 55 tok/s against 25.5 for llama.cpp. On a 40K-token document Strata did 382 tok/s prefill and 62 tok/s decode, against 157 and 24.7.

## Limits
- One run per engine.
- 100 + 80 questions: each score is roughly ±6 points (GSM8K) and ±10 (HumanEval+).
- One quant, one box.
- Strata ran with speculative decoding on (it is lossless in principle); llama.cpp ran without it.

## Credits
- Strata: Niko1221 and contributors.
- P100 patches: qr0n.
- Quant: ISTA-DASLab.
- Test sets: OpenAI GSM8K (MIT) and EvalPlus HumanEval+ (Apache-2.0).
- The Strata tip came from a viewer, d2tw4all.

Video: https://youtube.com/watch?v=zT09gR54e9w. Channel: Junk GPU Lab.
