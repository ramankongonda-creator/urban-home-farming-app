"""Turn a '# %%' / '# %% [md]' script into a Jupyter notebook and execute it."""
import sys, re, nbformat
from nbconvert.preprocessors import ExecutePreprocessor
src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
cells = []
for block in re.split(r"^# %%", text, flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    if head.strip() == "[md]":
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbformat.write(nb, out)
print("wrote", out, len(cells), "cells")
