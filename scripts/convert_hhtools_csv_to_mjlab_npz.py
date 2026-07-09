"""DEPRECATED — use ``hhtools convert csv-to-npz``.

CSV/PKL -> canonical mjlab NPZ conversion now lives in the single-source
:mod:`hhtools.dataconvert` package and is exposed via the CLI and the web
"数据转换" panel. The old per-robot adapter YAML remapping is superseded by
direct MJCF joint-name matching (the MJCF is the source of truth for joint
order, body tree, FK and collision geometry).

    Old:  python scripts/convert_hhtools_csv_to_mjlab_npz.py CLIP.csv --robot t1 ...
    New:  hhtools convert csv-to-npz CLIP.csv --mjcf ROBOT.xml --out OUT.npz

This shim forwards remaining arguments to ``hhtools convert csv-to-npz``.
"""

from __future__ import annotations

import sys

from hhtools.cli.convert import app


def main() -> None:
    print(
        "[deprecated] scripts/convert_hhtools_csv_to_mjlab_npz.py is replaced by "
        "`hhtools convert csv-to-npz` (hhtools/dataconvert/).",
        file=sys.stderr,
    )
    sys.argv = ["hhtools-convert", "csv-to-npz", *sys.argv[1:]]
    app()


if __name__ == "__main__":
    main()
