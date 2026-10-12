#!/usr/bin/env python3
"""manifest.jsonl -> a labeled grid PNG, for eyeballing a sweep at a glance.

Labels each thumbnail with only the fields that actually differ across the
manifest (diffing the resolved recipes), not the whole config. See
develop/DESIGN.md.

Usage:
    python develop/harness/contact_sheet.py --manifest develop/out/manifest.jsonl
"""
import argparse
import json
import os

from PIL import Image, ImageDraw


def load_records(manifest_path):
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten(d, prefix=""):
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, path)
        else:
            yield path, value


def varying_fields(records):
    flattened = [dict(flatten(r["recipe"])) for r in records]
    all_paths = set().union(*(f.keys() for f in flattened)) if flattened else set()
    varying = sorted(p for p in all_paths if len({f.get(p) for f in flattened}) > 1)
    return flattened, varying


def _truncate(value, max_chars=28):
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def label_for(flat_record, varying_paths):
    if not varying_paths:
        return ""
    return "\n".join(f"{p.split('.')[-1]}={_truncate(flat_record.get(p))}"
                      for p in varying_paths)


def build_sheet(records, out_path, columns, thumb_size):
    flattened, varying = varying_fields(records)
    label_h = 20 * (len(varying) + 1)
    cell_w, cell_h = thumb_size, thumb_size + label_h

    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (record, flat) in enumerate(zip(records, flattened)):
        col, row = i % columns, i // columns
        x, y = col * cell_w, row * cell_h
        thumb = Image.open(record["png"]).convert("RGB").resize((thumb_size, thumb_size))
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_size + 2), label_for(flat, varying), fill="black")

    sheet.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=256)
    args = parser.parse_args()

    records = [r for r in load_records(args.manifest) if r.get("png") and not r.get("error")]
    if not records:
        raise SystemExit(f"no successful runs with images in {args.manifest}")

    out_path = args.out or os.path.join(os.path.dirname(args.manifest), "contact_sheet.png")
    build_sheet(records, out_path, args.columns, args.thumb_size)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
