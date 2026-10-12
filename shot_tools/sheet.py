#!/usr/bin/env python3
"""Contact sheet of the latest non-rejected draft of every shot in a shot file: one labelled
thumbnail per image_id, for reviewing a batch at a glance.
-> <project>/out/sheets/<shot file stem>[_<n>].png (split every --per images)

Usage: python develop/<project>/tools/sheet.py develop/<project>/shots/<file>.yaml [--per 24]
"""
import argparse
import os
import sqlite3

import yaml
from PIL import Image, ImageDraw, ImageFont

import render

THUMB = 320
LABEL_H = 22
COLUMNS = 6


def latest_draft(image_id):
    with sqlite3.connect(render.MANIFEST) as db:
        row = db.execute("SELECT run_id, png FROM images WHERE image_id = ? AND status = 'draft' "
                         "ORDER BY run_id DESC LIMIT 1", (image_id,)).fetchone()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shots")
    parser.add_argument("--per", type=int, default=24)
    cli = parser.parse_args()

    with open(cli.shots) as f:
        ids = [shot["image_id"] for shot in yaml.safe_load(f)["shots"]]
    cells = [(image_id, latest_draft(image_id)) for image_id in ids]
    cells = [(image_id, row) for image_id, row in cells if row]
    font = ImageFont.load_default(size=15)
    out_dir = os.path.join(render.DRAFT_DIR, "sheets")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(cli.shots))[0]
    for page, start in enumerate(range(0, len(cells), cli.per)):
        chunk = cells[start:start + cli.per]
        rows = (len(chunk) + COLUMNS - 1) // COLUMNS
        sheet = Image.new("RGB", (COLUMNS * THUMB, rows * (THUMB + LABEL_H)), (20, 18, 16))
        draw = ImageDraw.Draw(sheet)
        for i, (image_id, (run_id, png)) in enumerate(chunk):
            x, y = (i % COLUMNS) * THUMB, (i // COLUMNS) * (THUMB + LABEL_H)
            thumb = Image.open(os.path.join(render.REPO_ROOT, png)).convert("RGB")
            thumb.thumbnail((THUMB, THUMB))
            sheet.paste(thumb, (x + (THUMB - thumb.width) // 2, y + (THUMB - thumb.height) // 2))
            draw.text((x + 4, y + THUMB + 2), f"{image_id}  r{run_id}", fill=(230, 220, 205), font=font)
        path = os.path.join(out_dir, f"{stem}_{page}.png")
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
