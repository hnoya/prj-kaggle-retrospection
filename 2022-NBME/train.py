import argparse

from module import train


def run(train_path: str, features_path: str, pt_notes_path: str):
    """訓練

    Args:
        train_path (str, optional): 訓練用データセットcsvパス. Defaults to Input.train_csv.
        features_path (str, optional): コンテキストデータセットcsvパス. Defaults to Input.features_csv.
        pt_notes_path (str, optional): 患者ノートデータセットのcsvパス. Defaults to Input.pt_notes_csv.
    """
    train(train_path, features_path, pt_notes_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./data/train.csv")
    parser.add_argument("--features-path", type=str, default="./data/features.csv")
    parser.add_argument("--pt-notes-path", type=str, default="./data/patient_notes.csv")
    args = parser.parse_args()

    run(
        train_path=args.train_path,
        features_path=args.features_path,
        pt_notes_path=args.pt_notes_path,
    )
