"""Print the STRUCTURE of the tracking files -- never their values.

Safe to share the output: it lists XML tag paths, attribute names and channel ids, and for the raw
file only how many groups and how many comma-separated fields each group has. No coordinates, no
names, no frame numbers.

    python scripts/inspect_schema.py

Answers the question that decides how much dead-ball logic we need to build: does the ATD's ball
group carry more than x,y,speed -- for example z, an alive/dead flag or an owning team?
"""

import collections
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import require  # noqa: E402

N_LINES = 2000  # enough to see every shape the raw file uses, far less than the whole file


def strip_ns(tag):
    return tag.split("}", 1)[-1]


def xml_structure(path):
    tree = ET.parse(path)
    paths = collections.Counter()
    attrs = collections.defaultdict(set)
    channel_ids = set()

    def walk(node, prefix):
        here = prefix + "/" + strip_ns(node.tag)
        paths[here] += 1
        attrs[here].update(node.attrib.keys())
        for key, value in node.attrib.items():
            # Channel ids are schema ("player12_x", "ball_z"), not data. Digits are collapsed so
            # nothing that looks like a slot index or an id survives.
            if "channel" in key.lower() or key.lower() in ("id", "name") and "Channel" in node.tag:
                channel_ids.add(re.sub(r"\d+", "#", value))
        for child in node:
            walk(child, here)

    walk(tree.getroot(), "")
    print("XML tag paths (count) and attribute names:")
    for tag_path in sorted(paths):
        print("  {:<90} x{:<6} {}".format(tag_path, paths[tag_path], sorted(attrs[tag_path])))
    print("\nchannel ids (digits collapsed to #):")
    for channel in sorted(channel_ids):
        print("  " + channel)


def raw_structure(path):
    shapes = collections.Counter()
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index >= N_LINES:
                break
            parts = line.rstrip("\n").split(":")
            shape = [len(parts)]
            for part in parts[1:]:
                groups = part.split(";")
                fields = sorted({len(group.split(",")) for group in groups})
                shape.append((len(groups), tuple(fields)))
            shapes[tuple(shape)] += 1
    print("\nraw file, first {} lines: (parts, (groups, fields-per-group)...) -> count".format(
        N_LINES))
    for shape, count in shapes.most_common():
        print("  {} -> {}".format(shape, count))


if __name__ == "__main__":
    meta, raw = require(["atd_meta", "atd_raw"])
    xml_structure(meta)
    raw_structure(raw)
