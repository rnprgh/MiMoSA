"""Remove execution results from a Jupyter notebook before publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_outputs(notebook_path: Path) -> tuple[int, int]:
    """Clear code-cell outputs and execution counts in place."""
    with notebook_path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    code_cell_count = 0
    output_count = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_cell_count += 1
        output_count += len(cell.get("outputs", []))
        cell["outputs"] = []
        cell["execution_count"] = None

    with notebook_path.open("w", encoding="utf-8") as notebook_file:
        json.dump(notebook, notebook_file, ensure_ascii=False, indent=1)
        notebook_file.write("\n")

    return code_cell_count, output_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    code_cell_count, output_count = strip_outputs(args.notebook)
    print(
        f"Cleared {output_count} outputs from "
        f"{code_cell_count} code cells in {args.notebook}."
    )


if __name__ == "__main__":
    main()
