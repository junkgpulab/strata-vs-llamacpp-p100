# Reproduce it yourself: every step we took

This is the exact path on our box: what we ran, in order, and the traps we hit.
Where we used a different tool copy than the one you would get, it says so.

## 0. The box
- **CPU and RAM:** Dell Precision T5810, Xeon E5-2690 v4 (14 cores, AVX2, no AVX-512), 78 GB DDR4.
- **GPUs:** 2x Tesla P100 PCIe 16 GB (sm_60), each on PCIe 3.0 x16. Power cap 175 W per card.
- **Disks:** hard disks only. The model files sit on an HDD (~230 MB/s sequential, ~1 MB/s random).
- **Software:** Ubuntu 24.04.5, kernel 6.8.0-147, NVIDIA driver 580.178.04, CUDA toolkit 12.9, gcc-13 13.3, cmake 3.28.
  - **CUDA 13 dropped Pascal.** You need a CUDA 12.x toolkit; the driver can be newer.
- `ulimit -l` (memlock) was 9.8 GB, so Strata could not mlock its tables. It still works (see step 3).

## 1. The model (both engines use the same files)
- **Files:** ISTA-DASLab `Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, folder `IQ3_XXS`. Two shards, 75.8 GB in total:
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (experts and weights)
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (the 28.8 GB per-layer embedding / lookup table)
- **RAM rule (applies to both engines):** with 78 GB of RAM, nothing else may sit in the page cache, or pages of the model get evicted. Decode then falls to 7-9 tok/s from HDD seeks. Before every run we:
  - dropped every other model file from the page cache (`posix_fadvise(DONTNEED)`)
  - read both shards once in full (`cat a b > /dev/null`, 567 s on our HDD)
  - checked 0 MB of disk reads per request

## 2. Strata, patched for the P100
```sh
git clone https://github.com/qr0n/p100-strata && cd p100-strata     # we used commit 8b1b0f5 (2026-11-02)
CUDA_ARCH=60-real ./build.sh                                        # engine -> strata/build-p100/strata (~6 min)
```
- **Source:** `strata/` is upstream Strata v0.2.2 (011c9e60) plus qr0n's patches 0001-0012. `tools/verify-source.sh` proves that.
- **Python for the server:** the server needs Strata's Python packages: `python -m venv .venv && .venv/bin/pip install -r strata/requirements.txt`. We used the Python bundled with Strata Terminal 0.1.47, which already had them.
- **Weight pack and MTP draft head:** built once with Strata's tools. We ran the copies from Strata Terminal 0.1.47. The same tools ship in `strata/tools/` of v0.2.2, but we did not test those copies.
  ```sh
  python strata/tools/iq_pack.py  --gguf <models>/...IQ3_XXS-00001-of-00002.gguf --out <packs>/iq3_xxs
  python strata/tools/mtp_fetch.py fetch --out <mtp>          # ~5.2 GB of MTP tensors from the original Qwen checkpoint
  python strata/tools/mtp_pack.py --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf
  python strata/tools/mtp_rt.py   --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt
  ```
- **Expert profile:** `strata/data/expert-profile.bin`. We used the identical file name from Terminal 0.1.47.

## 3. Serve Strata
- **Config:** `configs/strata-cfg.json`. Fill in your paths. It is qr0n's `strata-sc117-peer.json` with:
  - our pack, our model and our MTP head
  - `--max-context 65536`
  - the same env switches: `STRATA_ROUTE_TAIL_SKIP=0 STRATA_Q8_SM60=1 STRATA_GU_F16=2 STRATA_SM60_D20_SUB=1 STRATA_DOWN_F16=1 STRATA_B6_F16=1`
- **Start command:**
  ```sh
  cd p100-strata/strata
  STRATA_ENGINE_READY_S=3600 python -m serve.server --engine strata --config ../configs/strata-cfg.json \
      --host 127.0.0.1 --port 8096 --no-api-key
  ```
- **Trap 1: the start timeout.** On 78 GB + HDD the first start died. The 40 GiB expert arena loads through the file cache, which pushes the lookup table out of RAM. Patch 0001's "re-touch" then pulls it back with random HDD reads, and the server's 900 s READY timeout kills the engine. Fix:
  - set `STRATA_ENGINE_READY_S=3600`
  - the moment the engine log prints `loaded ... GiB at ... GiB/s`, read shard 2 sequentially yourself (`cat shard2 > /dev/null`, 11 min for us). The re-touch then finds it in RAM.
  - With an SSD or 96 GB+ of RAM you probably don't need either (untested).
- **Trap 2: the fleet lock.** The Python server inherits open file descriptors. If you start it inside `flock`, it keeps the lock after your script ends. Kill it explicitly when you are done.
- **When it is ready:** `/health` shows `"loaded": true`. On our box 14,138 of 24,576 experts were on the GPUs, with 33 GB of RAM still free.

## 4. llama.cpp
- **Which build:** we ran our own P100 build of llama.cpp (not linked in this repo).
  - Plain upstream llama.cpp runs the same model with the same flags (without the `GGML_CUDA_P100*` env vars), but it was slower on our box: about 19-20 tok/s instead of 25-27, measured 10-23/10-24 with `--moe-cache-mib 12288`.
  - It should answer the same questions the same way within normal numeric noise. That's our expectation; we did not test it.
- **Command:**
  ```sh
  GGML_CUDA_P100=1 GGML_CUDA_REGISTER_HOST=1 GGML_CUDA_P100_V2=1 LLAMA_MOE_CACHE_MIB_DEV=9500,11000 \
  llama-server -m <models>/...IQ3_XXS-00001-of-00002.gguf --host 127.0.0.1 --port 5899 -c 16384 \
      -t 14 -np 1 -fa on --jinja --fit on --fit-target 1024 -sm layer -b 1024 -ub 1024 -ctk f16 -ctv f16 --moe-cache-mib 12288
  ```

## 5. The benchmark
```sh
sh scripts/get-data.sh                                   # GSM8K test set + HumanEval+ v0.1.10 (OriginFmt)
python3 scripts/gen.py 5899 any                       out/llamacpp
python3 scripts/gen.py 8096 qwen3.8-flash-next-iq3_xxs out/strata
python3 scripts/score.py . out/llamacpp/gen.jsonl out/strata/gen.jsonl   # needs numpy; runs model-written code: use a sandbox
```
- **What `gen.py` sends:** GSM8K items 0-99, then HumanEval+ tasks 0-79. Temperature 0, thinking off (`chat_template_kwargs.enable_thinking=false`). Max 1,024 / 1,536 tokens. The prompts are in the script.
- **What `score.py` checks:** prints both scores and lists every question where the engines disagree.
- **How long:** with llama.cpp at ~26 tok/s, 180 questions took 42 min; Strata took about 15 min after loading.
- **Our raw answers:** in `results/`. Score them with `score.py` without running any model.

## 6. What we did NOT control
- **One run per engine.** Strata is not bit-identical between runs (its own docs, and Strata issue #2685).
- **Draft-ahead mode:** Strata ran with MTP speculative decoding on (spec 4); llama.cpp without.
- **Speed vs other processes:** the llama.cpp run saw some HDD reads (30,864 read operations), and a read-only monitor polled its `/slots` once a second. Both affect speed only, not the answers.
