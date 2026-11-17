"""Execute the lab notebooks headlessly, keeping the outputs.

    python tools/run_notebooks.py              # all thirteen labs, in order
    python tools/run_notebooks.py NLP_Lab5     # just one

A lab is one folder and may hold more than one notebook (Lab 13 ships two
parts); every notebook in the folder is executed in file-name order, each in
its own fresh kernel.

Run it with the environment's own interpreter, e.g.
    .venv\\Scripts\\python.exe tools\\run_notebooks.py
"""
import glob
import os
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Windows without Developer Mode cannot create symlinks, so huggingface_hub
# prints a long warning into every notebook that loads a model.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Lab 4 lemmatises 30k reviews, Lab 5 trains Word2Vec on 158k Simpsons lines, and
# Lab 10 trains a translation model for 12 epochs, so those need room. Lab 12
# pulls 440 MB of bert-base-uncased on a cold cache. The rest finish in under a
# minute.
TIMEOUTS = {"NLP_Lab10": 3600, "NLP_Lab7": 3600, "NLP_Lab5": 3600,
            "NLP_Lab3": 2400, "NLP_Lab4": 2400, "NLP_Lab12": 1800}

wanted = sys.argv[1:] or [f"NLP_Lab{i}" for i in range(1, 14)]

failed = 0
for lab in wanted:
    hits = sorted(glob.glob(os.path.join(ROOT, lab, "*.ipynb")))
    if not hits:
        print(f"{lab:<10} SKIPPED: no .ipynb found")
        failed += 1
        continue

    for path in hits:
        name = lab if len(hits) == 1 else f"{lab}/{os.path.basename(path)}"

        nb = nbformat.read(path, as_version=4)
        started = time.time()
        client = NotebookClient(
            nb,
            timeout=TIMEOUTS.get(lab, 900),
            kernel_name="nlp-labs",
            allow_errors=False,
            resources={"metadata": {"path": os.path.dirname(path)}},
        )
        try:
            client.execute()
            status = "OK"
        except Exception as exc:
            status = f"FAILED: {type(exc).__name__}: {str(exc)[:300]}"
            failed += 1
            traceback.print_exc(limit=2)

        nbformat.write(nb, path)
        errors = [i for i, c in enumerate(nb.cells)
                  if c.cell_type == "code"
                  and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
        print(f"{name:<30} {status:<12} {time.time() - started:7.1f}s  errors in cells: {errors}",
              flush=True)

sys.exit(1 if failed else 0)
