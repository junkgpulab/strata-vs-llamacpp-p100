#!/usr/bin/env python3
# FORGE: standard-benchmark generations against an OpenAI-compatible server. Same prompts, greedy, thinking off, for both
# engines. GSM8K test items 0-99 (openai/grade-school-math test.jsonl), HumanEval+ tasks 0-79 (evalplus v0.1.10 OriginFmt).
# Usage: gen.py PORT MODEL OUTDIR
import json, sys, time, urllib.request, os
port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
H = os.path.dirname(os.path.abspath(__file__))
def ask(content, max_tokens):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
    r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(r, timeout=600)); return d['choices'][0]['message'].get('content') or '', d.get('timings', {})
jobs = [('gsm8k', i, json.loads(l)) for i, l in enumerate(open(f'{H}/gsm8k-test.jsonl')) if i < 100] + \
       [('hep', i, json.loads(l)) for i, l in enumerate(open(f'{H}/HEP.jsonl')) if i < 80]
with open(f'{out}/gen.jsonl', 'a') as f:
    for kind, i, x in jobs:
        if kind == 'gsm8k':
            q = x['question'] + '\n\nSolve the problem step by step. Put the final answer on the last line as: The answer is <number>.'
            mt = 1024
        else:
            q = ('Complete the following Python function. Reply with the complete function (including the signature and any '
                 'imports it needs) in one ```python code block.\n\n```python\n' + x['prompt'] + '```')
            mt = 1536
        t0 = time.time()
        try: a, t = ask(q, mt)
        except Exception as e: a, t = f'ERROR {e!r}', {}
        f.write(json.dumps({'kind': kind, 'i': i, 'answer': a, 'tps': t.get('predicted_per_second'), 'n': t.get('predicted_n'),
                            'wall': round(time.time() - t0, 1)}) + '\n'); f.flush()
        print(kind, i, t.get('predicted_per_second'), t.get('predicted_n'), flush=True)
