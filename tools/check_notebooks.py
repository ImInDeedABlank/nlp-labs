"""Sanity-check the nine lab notebooks without executing them.

Checks, for every cell:
  - every code cell compiles (ignoring IPython-only lines: %magic, !shell, `name?`)
  - every cell has a unique id, which nbformat 4.5 requires
  - no cell holds an error output
  - no non-empty code cell is left unexecuted

    python tools/check_notebooks.py
"""
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HELP_LINE = re.compile(r"^[\w\.\[\]\(\)\"']+\?\??\s*$")


def check(path):
    nb = json.load(open(path, encoding="utf-8"))
    cells = nb["cells"]
    problems = []

    ids = [c.get("id") for c in cells]
    if not all(ids) or len(set(ids)) != len(ids):
        problems.append("missing or duplicate cell ids")

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])

        if src.strip():
            clean = "\n".join(
                "" if ln.lstrip().startswith(("%", "!", "?")) or HELP_LINE.match(ln.strip())
                else ln
                for ln in src.splitlines()
            )
            try:
                compile(clean, f"cell {i}", "exec")
            except SyntaxError as e:
                problems.append(f"cell {i}: SyntaxError line {e.lineno}: {e.msg}")

            if c.get("execution_count") is None:
                problems.append(f"cell {i}: not executed")

        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                problems.append(f"cell {i}: {o.get('ename')}: {o.get('evalue')}")

    return cells, problems


bad = 0
for p in sorted(glob.glob(os.path.join(ROOT, "NLP_Lab*", "*.ipynb"))):
    lab = os.path.basename(os.path.dirname(p))
    cells, problems = check(p)
    code = sum(1 for c in cells if c["cell_type"] == "code")
    bad += 1 if problems else 0
    print(f"{lab:<10} {'OK ' if not problems else 'BAD'}  {len(cells):>3} cells ({code} code)")
    for problem in problems:
        print(f"    {problem}")

print(f"\n{9 - bad}/9 notebooks clean")
sys.exit(1 if bad else 0)
