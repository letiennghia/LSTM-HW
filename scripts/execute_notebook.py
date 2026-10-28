"""Execute a verification copy with the current interpreter in a fresh kernel."""

import argparse
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--notebook", default="imdb_sentiment.ipynb", type=Path)
    parser.add_argument("--timeout", default=3600, type=int, help="Per-cell execution timeout in seconds")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache" / "jupyter-runtime"))
    notebook = nbformat.read(root / args.notebook, as_version=4)
    nbformat.validate(notebook)
    settings = [cell for cell in notebook.cells if cell.cell_type == "code" and 'MODE = "full"' in cell.source]
    if len(settings) != 1:
        raise ValueError("Expected exactly one notebook mode setting")
    settings[0].source = settings[0].source.replace('MODE = "full"', f'MODE = "{args.mode}"')
    output = root / "outputs" / "verification" / f"{args.notebook.stem}_{args.mode}.executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=args.timeout, resources={"metadata": {"path": str(root)}})
    client.on_cell_start = lambda cell, cell_index, **kwargs: print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        print(f"Executed notebook saved: {output}", flush=True)
    nbformat.validate(notebook)
    assert not any(out.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs)
    print("Notebook execution completed without cell errors.", flush=True)


if __name__ == "__main__":
    main()
