# 2022 / NBME / 患者のメモ書き
- kaggle / [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)

## モジュール
- 参考にしたNotebook: [NBME / Deberta-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train) / [NBME / Deberta-base baseline [inference]](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference/notebook) / [Author's GitHub](https://github.com/YasufumiNakama)

## 使い方
1. zipでコードをダウンロードする.
2. colab上で`!python train.py`で訓練
3. colab上で`!python test.py`で推論
- kaggleで使う場合は、`from .train import main`で関数を呼び出して使う.
