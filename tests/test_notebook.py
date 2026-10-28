import ast
import nbformat


def test_notebook_valid_and_code_compiles():
    notebook = nbformat.read("imdb_sentiment.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert sum('MODE = "full"' in source for source in code) == 1
    for source in code:
        ast.parse(source)
    assert "experiment.train" in "\n".join(code)
