import argparse

from module import test


def run(test_path: str, features_path: str, pt_notes_path: str):
    """訓練

    Args:
        test_path (str, optional): 訓練用データセットcsvパス. Defaults to Input.test_csv.
        features_path (str, optional): コンテキストデータセットcsvパス. Defaults to Input.features_csv.
        pt_notes_path (str, optional): 患者ノートデータセットのcsvパス. Defaults to Input.pt_notes_csv.
    """
    test(test_path, features_path, pt_notes_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, default="./data/test.csv")
    parser.add_argument("--features-path", type=str, default="./data/features.csv")
    parser.add_argument("--pt-notes-path", type=str, default="./data/patient_notes.csv")
    args = parser.parse_args()

    run(
        test_path=args.test_path,
        features_path=args.features_path,
        pt_notes_path=args.pt_notes_path,
    )
