"""DEPRECATED — use the 数据转换 web panel or ``hhtools convert check``.

Interactive replay + penetration/contact visualisation moved into the browser
"数据转换" panel (in-browser three.js with server-side MuJoCo FK + collision /
contact-force overlay). The headless penetration audit now lives in the
single-source :mod:`hhtools.dataconvert.contacts` package.

    Old:  python scripts/preview_npz.py CLIP.npz --check-penetration
    New (headless):  hhtools convert check CLIP.npz --mjcf ROBOT.xml
    New (visual):    hhtools web  ->  侧边栏「数据转换」面板

This shim forwards remaining arguments to ``hhtools convert check``.
"""

from __future__ import annotations

import sys

from hhtools.cli.convert import app


def main() -> None:
    print(
        "[deprecated] scripts/preview_npz.py is replaced by the 数据转换 web panel "
        "and `hhtools convert check` (hhtools/dataconvert/).",
        file=sys.stderr,
    )
    sys.argv = ["hhtools-convert", "check", *sys.argv[1:]]
    app()


if __name__ == "__main__":
    main()
