"""Checks that every phase notebook carries the same copy of the shared code.

Kaggle needs self-contained notebooks, so the data loader and the training loop are copied into each one.
This script fails if any copy differs, so a fix applied to one notebook cannot silently miss the others.
Usage: python experiments/check_notebooks.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def code_cells(path):
    return [''.join(c['source']) for c in json.load(open(path, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']


def loader_core(cell):
    core = cell[cell.index('def load_dataset'):cell.index('def split_centers')]
    # Phase 4 adds a raw-band preprocessing function on purpose; everything else must match.
    return re.sub(r'def preprocess_raw\(img\):.*?(?=def preprocess\(img\):)', '', core, flags=re.S)


notebooks = sorted(glob.glob(os.path.join(HERE, 'phase*.ipynb')))
assert len(notebooks) >= 2, notebooks
parts = {'data loader (cell 2)': lambda c: loader_core(c[1]), 'training loop (cell 4)': lambda c: c[3]}
cells = {nb: code_cells(nb) for nb in notebooks}
ok = True
for name, get in parts.items():
    ref_nb = notebooks[0]
    ref = get(cells[ref_nb])
    for nb in notebooks[1:]:
        if get(cells[nb]) != ref:
            ok = False
            print(f'DIFFERS: {name} in {os.path.basename(nb)} vs {os.path.basename(ref_nb)}')
assert "if len(idx) < 2:" in parts['training loop (cell 4)'](cells[notebooks[0]]), 'batch-of-one guard missing'
assert ok, 'shared notebook code differs; apply the same change to every phase notebook'
print(f'OK: {len(notebooks)} notebooks share the same data loader and training loop')
