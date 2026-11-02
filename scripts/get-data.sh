#!/bin/sh
# The two public test sets (not redistributed here). We use GSM8K test items 0-99 and HumanEval+ tasks 0-79.
curl -sSfL -o gsm8k-test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -sSfL https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz | gunzip > HEP.jsonl
