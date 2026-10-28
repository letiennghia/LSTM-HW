# LSTM Comparision against Vanilla RNN and other SML methods

Both notebooks retain their executed full-experiment outputs:
- imdb_sentiment.ipynb: original Logistic Regression, SVM, RNN and LSTM benchmark.
- imdb_lstm_optimization.ipynb: 16 LSTM ablations and validation-selected final test.

## Local execution

1. Extract this ZIP into a folder. Keep src/, configs/, scripts/ and outputs/ together with the notebooks.
2. Install Python 3.11 and create/activate a virtual environment if desired.
3. Open a terminal in the extracted folder and run:

   python -m pip install -r requirements.txt
   python -m jupyterlab

4. Open either notebook and choose Run All. The saved results can be reviewed without rerunning.
5. For a quick execution check, change MODE = "full" to MODE = "smoke" in the notebook.

The notebooks import code from src/imdb_sentiment; they are not standalone files.
All required project modules, both YAML configurations, packaging files, dependencies,
tests and the fresh-kernel execution script are included. The optimization notebook's
small historical benchmark reference files are included at the paths expected by its config.

## Google Colab

Open either .ipynb in Colab and select a GPU runtime. Run the setup cell and upload this
same ZIP when prompted. The archive has project files directly at its root, as required
by the notebooks' upload/extraction code. Run the remaining cells in order.

## Data and execution

Python dependencies and the public stanfordnlp/imdb dataset require Internet access on
first use. CUDA is used when available; CPU fallback is supported. Full optimization took
about 71 minutes of aggregate candidate training on the verified RTX 3060 environment.
The official 20k/5k/25k split and validation-only selection are preserved.

No virtual environments, data caches or model checkpoints are bundled: running the notebooks
trains the models and creates new run directories. Saved notebook outputs, summary reports
and comparison CSVs are included for review. Do not use the bundled historical summary
folders with the evaluate command: use the new study directory printed by a fresh run.

See README.md and README_OPTIMIZATION.md for detailed methodology and commands.
SHA256SUMS.txt provides integrity hashes for the bundled project files.
