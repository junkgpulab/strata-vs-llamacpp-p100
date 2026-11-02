#!/usr/bin/env python3
# STUDIO: score gen.jsonl. GSM8K: the number after "The answer is" (else the last number) vs the gold after "####".
# HumanEval+: the first ```python block (prompt prepended if the function is missing) + the task's plus tests, run in a
# fresh python3 with a 30 s timeout in a temp dir. Usage: score.py DATA_DIR GEN.jsonl [GEN2.jsonl ...]
import json, re, subprocess, sys, tempfile, os
dd = sys.argv[1]
gsm = [json.loads(l) for l in open(f'{dd}/gsm8k-test.jsonl')]; hep = [json.loads(l) for l in open(f'{dd}/HEP.jsonl')]
def num(s):
    s = s.replace(',', '').replace('$', '')
    m = re.findall(r'answer is\s*:?\s*\**\s*(-?\d+(?:\.\d+)?)', s, re.I) or re.findall(r'-?\d+(?:\.\d+)?', s)
    return float(m[-1]) if m else None
def code(a, x):
    m = re.findall(r'```(?:python)?\n(.*?)```', a, re.S); c = m[0] if m else a
    return c if f"def {x['entry_point']}" in c else x['prompt'] + c
res = {}
for g in sys.argv[2:]:
    rows = [json.loads(l) for l in open(g)]; ok = {'gsm8k': [], 'hep': []}
    for r in rows:
        if r['kind'] == 'gsm8k':
            gold = float(gsm[r['i']]['answer'].split('####')[-1].strip().replace(',', ''))
            ok['gsm8k'].append((r['i'], num(r['answer']) == gold))
        else:
            x = hep[r['i']]; prog = code(r['answer'], x) + '\n\n' + x['test'] + f"\n\ncheck({x['entry_point']})\n"
            with tempfile.TemporaryDirectory() as td:
                open(f'{td}/p.py', 'w').write(prog)
                try: p = subprocess.run([sys.executable, '-I', 'p.py'], cwd=td, capture_output=True, timeout=30); good = p.returncode == 0
                except subprocess.TimeoutExpired: good = False
            ok['hep'].append((r['i'], good))
    res[g] = ok
    for k, v in ok.items():
        tps = [r['tps'] for r in rows if r['kind'] == k and r['tps']]
        print(f"{g} {k}: {sum(b for _, b in v)}/{len(v)} right, median decode {sorted(tps)[len(tps)//2] if tps else '-'} t/s")
if len(res) == 2:
    a, b = list(res.values())
    for k in ('gsm8k', 'hep'):
        da = dict(a[k]); db = dict(b[k]); onlyA = [i for i in da if da[i] and not db.get(i)]; onlyB = [i for i in db if db[i] and not da.get(i)]
        print(f"{k}: right only in 1st: {len(onlyA)} {onlyA}; right only in 2nd: {len(onlyB)} {onlyB}")
