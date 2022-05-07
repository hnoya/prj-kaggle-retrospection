# 2022 / NBME / 患者のメモ書き
- kaggle / [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)

## モジュール
- 参考にしたNotebook:
    - 訓練: [NBME / Deberta-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train)
    - 予測: [NBME / Deberta-base baseline [inference]](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference/notebook)
    - 高速化: [Fast inference by padding optimization](https://www.kaggle.com/code/anyai28/fast-inference-by-padding-optimization)

## 使い方
### Google Colabの場合
- google_colab.ipynb参考

### Kaggleの場合
- !python train.py実行時にコマンドラインでCSVパスを渡す
- あとは概ねGoogle Colabと同様

## その他
- このコードは使用していたNotebooksをリファクタリングしたもので、パラメータ・学習方法等は実際のものと異なる
    - csv
        - 最終Submitの訓練時は全てPseudo Labeling (Hard)後のcsvを使用
        - Pseudo Labelingは、通常学習→OOFのEnsemble→未Label部分を予測して作成・使用した
    - Epoch
        - Pseudo Labeling前はlarge系: 5 / base系: 10
        - Pseudo Labeling後はEpochは1
    - batch size
        - A100, V100を主に使用しており、16
    - seed
        - 適宜変更していた
