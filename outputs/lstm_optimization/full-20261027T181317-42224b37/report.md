# LSTM optimization study

Full 20k/5k validation study, seed 42.

| experiment_id | status | val_accuracy | val_f1 | val_loss | training_seconds | best_epoch | delta_accuracy_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01_original_control | complete | 0.8452 | 0.8514 | 0.3960 | 53.7118 | 5 | -3.4200 |
| 02_anchor | complete | 0.8794 | 0.8758 | 0.2795 | 329.1722 | 15 | 0.0000 |
| 03_vocab_20000 | complete | 0.8768 | 0.8820 | 0.2915 | 327.5208 | 14 | -0.2600 |
| 04_length_300 | complete | 0.8714 | 0.8687 | 0.3035 | 253.6196 | 12 | -0.8000 |
| 05_length_500 | complete | 0.8840 | 0.8850 | 0.2819 | 382.5686 | 15 | 0.4600 |
| 06_head_tail | complete | 0.8804 | 0.8814 | 0.2800 | 323.3310 | 15 | 0.1000 |
| 07_unidirectional | complete | 0.8666 | 0.8679 | 0.3211 | 167.9347 | 8 | -1.2800 |
| 08_final_pooling | complete | 0.8658 | 0.8596 | 0.3219 | 213.5752 | 14 | -1.3600 |
| 09_max_pooling | complete | 0.8842 | 0.8839 | 0.2791 | 316.5378 | 14 | 0.4800 |
| 10_no_embedding_dropout | complete | 0.8784 | 0.8766 | 0.3134 | 323.5270 | 14 | -0.1000 |
| 11_classifier_dropout_03 | complete | 0.8792 | 0.8793 | 0.2902 | 323.3777 | 14 | -0.0200 |
| 12_word_dropout | complete | 0.8862 | 0.8890 | 0.2825 | 324.8152 | 15 | 0.6800 |
| 13_no_weight_decay | complete | 0.8666 | 0.8706 | 0.3464 | 216.6483 | 6 | -1.2800 |
| 14_exponential_scheduler | complete | 0.8682 | 0.8685 | 0.3100 | 320.3695 | 15 | -1.1200 |
| 15_learning_rate_001 | complete | 0.8802 | 0.8822 | 0.2820 | 213.6078 | 6 | 0.0800 |
| 16_short_budget | complete | 0.8578 | 0.8575 | 0.3325 | 170.7553 | 7 | -2.1600 |

Frozen winner: **12_word_dropout**. Ranking: validation accuracy descending, F1 descending, loss ascending, experiment ID ascending. Checkpoints are restored by validation loss; no test metric selected configurations.

## Observed ablations

- Vocabulary 20000: reduced validation accuracy relative to the anchor (-0.26 percentage points); F1 change +0.0062, loss change +0.0120; training 327.52s.
- Sequence length 300: reduced validation accuracy relative to the anchor (-0.80 percentage points); F1 change -0.0070, loss change +0.0240; training 253.62s.
- Sequence length 500: improved validation accuracy relative to the anchor (+0.46 percentage points); F1 change +0.0092, loss change +0.0024; training 382.57s.
- Head-tail truncation: improved validation accuracy relative to the anchor (+0.10 percentage points); F1 change +0.0056, loss change +0.0005; training 323.33s.
- Unidirectional recurrence: reduced validation accuracy relative to the anchor (-1.28 percentage points); F1 change -0.0079, loss change +0.0416; training 167.93s.
- Final-state pooling: reduced validation accuracy relative to the anchor (-1.36 percentage points); F1 change -0.0162, loss change +0.0424; training 213.58s.
- Masked max pooling: improved validation accuracy relative to the anchor (+0.48 percentage points); F1 change +0.0081, loss change -0.0004; training 316.54s.
- Embedding dropout disabled: reduced validation accuracy relative to the anchor (-0.10 percentage points); F1 change +0.0008, loss change +0.0339; training 323.53s.
- Classifier dropout 0.3: reduced validation accuracy relative to the anchor (-0.02 percentage points); F1 change +0.0035, loss change +0.0107; training 323.38s.
- Word dropout 0.1: improved validation accuracy relative to the anchor (+0.68 percentage points); F1 change +0.0132, loss change +0.0030; training 324.82s.
- Weight decay disabled: reduced validation accuracy relative to the anchor (-1.28 percentage points); F1 change -0.0052, loss change +0.0669; training 216.65s.
- Exponential LR decay gamma 0.95: reduced validation accuracy relative to the anchor (-1.12 percentage points); F1 change -0.0073, loss change +0.0305; training 320.37s.
- Learning rate 0.001: improved validation accuracy relative to the anchor (+0.08 percentage points); F1 change +0.0065, loss change +0.0025; training 213.61s.
- Maximum 8 epochs and patience 2: reduced validation accuracy relative to the anchor (-2.16 percentage points); F1 change -0.0183, loss change +0.0531; training 170.76s.

The original control versus anchor changes a bundle of settings. The short-budget experiment changes both epochs and patience. Other ablations change one listed setting. Single-seed differences are descriptive, not significance tests; feature interactions are not measured.

Total completed candidate training time: 4261.07s, excluding preprocessing, selection and final test.

## Final test

| model | status | accuracy | f1 | macro_f1 | training_seconds |
| --- | --- | --- | --- | --- | --- |
| logreg | historical benchmark | 0.9023 | 0.9023 | 0.9023 | 3.4866 |
| svm | historical benchmark | 0.9009 | 0.9006 | 0.9009 | 3.5356 |
| rnn | historical benchmark | 0.7526 | 0.7420 | 0.7521 | 78.0468 |
| lstm | historical benchmark | 0.8403 | 0.8460 | 0.8401 | 87.9016 |
| selected:12_word_dropout | full | 0.8853 | 0.8886 | 0.8852 | 324.8152 |

Historical results are unchanged, on the same split. They were read for reporting only after selection and final evaluation. Timings compare different architectures/devices and exclude preprocessing.

## Research basis and departures

[Hameed & Garcia-Zapirain (2020)](https://doi.org/10.1109/ACCESS.2020.2988550) motivates BiLSTM mean/max pooling, regularization and evaluating 400 tokens; its GloVe embeddings and cross-validation differ from this study. [Sachan et al. (arXiv:2009.04007)](https://arxiv.org/abs/2009.04007) motivates max pooling, embedding/word dropout, vocabulary ablations and learning-rate decay; its pretrained embeddings and semi-supervised objectives are excluded here. The [Keras example](https://keras.io/examples/nlp/bidirectional_lstm_imdb/) illustrates bidirectional recurrence but uses a different validation protocol. Here embeddings are randomly initialized and trained only on 20k labeled training reviews; the official test split is never used for tuning. Head-tail truncation, exact dropout values, decay gamma and other starting hyperparameters are local hypotheses, not established optima. Published results are not directly reproducible under this repository's split or protocol.
