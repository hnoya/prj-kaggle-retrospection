import ast
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from .config import Config, feature_col, train_col


def clean_spaces(txt: str) -> str:
    """テキストの改行等を空白に変換する
    # TODO: Tokenizer側で\n等を考慮する

    Args:
        txt (str): 改行を含むテキスト

    Returns:
        str: 改行を含まないテキスト
    """
    txt = re.sub("\n", " ", txt)
    txt = re.sub("\t", " ", txt)
    txt = re.sub("\r", " ", txt)
    return txt


def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
    """誤ったアノテーションを修正する

    Args:
        features (pd.DataFrame): 誤ったアノテーションが含まれるデータフレーム

    Returns:
        pd.DataFrame: アノテーションを修正したデータフレーム
    """
    features.loc[27, feature_col.feature_text] = "Last-Pap-smear-1-year-ago"
    return features


def fix_train_annotation(train: pd.DataFrame) -> pd.DataFrame:
    """誤ったアノテーションを修正する

    Args:
        train (pd.DataFrame): 誤ったアノテーションが含まれる訓練データフレーム

    Returns:
        pd.DataFrame: アノテーションを修正した訓練データフレーム
    """
    train.loc[338, "annotation"] = ast.literal_eval('[["father heart attack"]]')
    train.loc[338, "location"] = ast.literal_eval('[["764 783"]]')

    train.loc[621, "annotation"] = ast.literal_eval('[["for the last 2-3 months"]]')
    train.loc[621, "location"] = ast.literal_eval('[["77 100"]]')

    train.loc[655, "annotation"] = ast.literal_eval(
        '[["no heat intolerance"], ["no cold intolerance"]]'
    )
    train.loc[655, "location"] = ast.literal_eval(
        '[["285 292;301 312"], ["285 287;296 312"]]'
    )

    train.loc[1262, "annotation"] = ast.literal_eval('[["mother thyroid problem"]]')
    train.loc[1262, "location"] = ast.literal_eval('[["551 557;565 580"]]')

    train.loc[1265, "annotation"] = ast.literal_eval(
        "[['felt like he was going to \"pass out\"']]"
    )
    train.loc[1265, "location"] = ast.literal_eval('[["131 135;181 212"]]')

    train.loc[1396, "annotation"] = ast.literal_eval('[["stool , with no blood"]]')
    train.loc[1396, "location"] = ast.literal_eval('[["259 280"]]')

    train.loc[1591, "annotation"] = ast.literal_eval('[["diarrhoe non blooody"]]')
    train.loc[1591, "location"] = ast.literal_eval('[["176 184;201 212"]]')

    train.loc[1615, "annotation"] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
    train.loc[1615, "location"] = ast.literal_eval('[["249 257;271 288"]]')

    train.loc[1664, "annotation"] = ast.literal_eval('[["no vaginal discharge"]]')
    train.loc[1664, "location"] = ast.literal_eval('[["822 824;907 924"]]')

    train.loc[1714, "annotation"] = ast.literal_eval(
        '[["started about 8-10 hours ago"]]'
    )
    train.loc[1714, "location"] = ast.literal_eval('[["101 129"]]')

    train.loc[1929, "annotation"] = ast.literal_eval('[["no blood in the stool"]]')
    train.loc[1929, "location"] = ast.literal_eval('[["531 539;549 561"]]')

    train.loc[2134, "annotation"] = ast.literal_eval(
        '[["last sexually active 9 months ago"]]'
    )
    train.loc[2134, "location"] = ast.literal_eval('[["540 560;581 593"]]')

    train.loc[2191, "annotation"] = ast.literal_eval('[["right lower quadrant pain"]]')
    train.loc[2191, "location"] = ast.literal_eval('[["32 57"]]')

    train.loc[2553, "annotation"] = ast.literal_eval('[["diarrhoea no blood"]]')
    train.loc[2553, "location"] = ast.literal_eval('[["308 317;376 384"]]')

    train.loc[3124, "annotation"] = ast.literal_eval('[["sweating"]]')
    train.loc[3124, "location"] = ast.literal_eval('[["549 557"]]')

    train.loc[3858, "annotation"] = ast.literal_eval(
        '[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]'  # noqa
    )
    train.loc[3858, "location"] = ast.literal_eval(
        '[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]'
    )

    train.loc[4373, "annotation"] = ast.literal_eval('[["for 2 months"]]')
    train.loc[4373, "location"] = ast.literal_eval('[["33 45"]]')

    train.loc[4763, "annotation"] = ast.literal_eval('[["35 year old"]]')
    train.loc[4763, "location"] = ast.literal_eval('[["5 16"]]')

    train.loc[4782, "annotation"] = ast.literal_eval('[["darker brown stools"]]')
    train.loc[4782, "location"] = ast.literal_eval('[["175 194"]]')

    train.loc[4908, "annotation"] = ast.literal_eval('[["uncle with peptic ulcer"]]')
    train.loc[4908, "location"] = ast.literal_eval('[["700 723"]]')

    train.loc[6016, "annotation"] = ast.literal_eval('[["difficulty falling asleep"]]')
    train.loc[6016, "location"] = ast.literal_eval('[["225 250"]]')

    train.loc[6192, "annotation"] = ast.literal_eval(
        '[["helps to take care of aging mother and in-laws"]]'
    )
    train.loc[6192, "location"] = ast.literal_eval('[["197 218;236 260"]]')

    train.loc[6380, "annotation"] = ast.literal_eval(
        '[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]'
    )
    train.loc[6380, "location"] = ast.literal_eval(
        '[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]'  # noqa
    )

    train.loc[6562, "annotation"] = ast.literal_eval(
        '[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]'
    )
    train.loc[6562, "location"] = ast.literal_eval(
        '[["290 320;327 337"], ["290 320;342 358"]]'
    )

    train.loc[6862, "annotation"] = ast.literal_eval(
        '[["stressor taking care of many sick family members"]]'
    )
    train.loc[6862, "location"] = ast.literal_eval('[["288 296;324 363"]]')

    train.loc[7022, "annotation"] = ast.literal_eval(
        '[["heart started racing and felt numbness for the 1st time in her finger tips"]]'
    )
    train.loc[7022, "location"] = ast.literal_eval('[["108 182"]]')

    train.loc[7422, "annotation"] = ast.literal_eval('[["first started 5 yrs"]]')
    train.loc[7422, "location"] = ast.literal_eval('[["102 121"]]')

    train.loc[8876, "annotation"] = ast.literal_eval('[["No shortness of breath"]]')
    train.loc[8876, "location"] = ast.literal_eval('[["481 483;533 552"]]')

    train.loc[9027, "annotation"] = ast.literal_eval(
        '[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]'
    )
    train.loc[9027, "location"] = ast.literal_eval('[["92 102"], ["123 164"]]')

    train.loc[9938, "annotation"] = ast.literal_eval(
        '[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]'
    )
    train.loc[9938, "location"] = ast.literal_eval(
        '[["89 117"], ["122 138"], ["368 402"]]'
    )

    train.loc[9973, "annotation"] = ast.literal_eval('[["gaining 10-15 lbs"]]')
    train.loc[9973, "location"] = ast.literal_eval('[["344 361"]]')

    train.loc[10513, "annotation"] = ast.literal_eval(
        '[["weight gain"], ["gain of 10-16lbs"]]'
    )
    train.loc[10513, "location"] = ast.literal_eval('[["600 611"], ["607 623"]]')

    train.loc[11551, "annotation"] = ast.literal_eval(
        '[["seeing her son knows are not real"]]'
    )
    train.loc[11551, "location"] = ast.literal_eval('[["386 400;443 461"]]')

    train.loc[11677, "annotation"] = ast.literal_eval(
        '[["saw him once in the kitchen after he died"]]'
    )
    train.loc[11677, "location"] = ast.literal_eval('[["160 201"]]')

    train.loc[12124, "annotation"] = ast.literal_eval(
        '[["tried Ambien but it didnt work"]]'
    )
    train.loc[12124, "location"] = ast.literal_eval('[["325 337;349 366"]]')

    train.loc[12279, "annotation"] = ast.literal_eval(
        '[["heard what she described as a party later than evening these things did not actually happen"]]'
    )
    train.loc[12279, "location"] = ast.literal_eval('[["405 459;488 524"]]')

    train.loc[12289, "annotation"] = ast.literal_eval(
        '[["experienced seeing her son at the kitchen table these things did not actually happen"]]'
    )
    train.loc[12289, "location"] = ast.literal_eval('[["353 400;488 524"]]')

    train.loc[13238, "annotation"] = ast.literal_eval(
        '[["SCRACHY THROAT"], ["RUNNY NOSE"]]'
    )
    train.loc[13238, "location"] = ast.literal_eval('[["293 307"], ["321 331"]]')

    train.loc[13297, "annotation"] = ast.literal_eval(
        '[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]'
    )
    train.loc[13297, "location"] = ast.literal_eval(
        '[["182 221"], ["182 213;225 234"]]'
    )

    train.loc[13299, "annotation"] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
    train.loc[13299, "location"] = ast.literal_eval('[["79 88"], ["409 418"]]')

    train.loc[13845, "annotation"] = ast.literal_eval(
        '[["headache global"], ["headache throughout her head"]]'
    )
    train.loc[13845, "location"] = ast.literal_eval(
        '[["86 94;230 236"], ["86 94;237 256"]]'
    )

    train.loc[14083, "annotation"] = ast.literal_eval(
        '[["headache generalized in her head"]]'
    )
    train.loc[14083, "location"] = ast.literal_eval('[["56 64;156 179"]]')
    return train


def add_validation_id(df: pd.DataFrame) -> pd.DataFrame:
    """バリデーションのfold IDを付与する

    Args:
        df (pd.DataFrame): fold IDが付与されていないデータフレーム

    Returns:
        pd.DataFrame: fold IDが付与されたデータフレーム
    """
    foldmaker = StratifiedGroupKFold(
        n_splits=Config.n_fold, shuffle=True, random_state=Config.seed
    )
    for fold_idx, (train_idx, valid_idx) in enumerate(
        foldmaker.split(df, df[train_col.location], df[train_col.pn_num])
    ):
        df.loc[valid_idx, train_col.fold] = fold_idx
    df[train_col.fold] = df[train_col.fold].astype(int)
    return df


def load_train_df(
    train_path: str, features_path: str, pt_notes_path: str
) -> pd.DataFrame:
    """訓練用データフレームを読み込む

    Args:
        train_path (str): 訓練用データCSVのパス
        features_path (str): コンテキスト文章のCSVのパス
        pt_notes_path (str): 患者ノート文章のCSVのパス

    Returns:
        pd.DataFrame: 訓練用データフレーム
    """
    train = pd.read_csv(train_path)
    train[train_col.annotation] = train[train_col.annotation].apply(ast.literal_eval)
    train[train_col.location] = train[train_col.location].apply(ast.literal_eval)
    features = pd.read_csv(features_path)
    features = preprocess_features(features)
    patient_notes = pd.read_csv(pt_notes_path)
    train = train.merge(
        features, on=[train_col.feature_num, train_col.case_num], how="left"
    )
    train = train.merge(
        patient_notes, on=[train_col.pn_num, train_col.case_num], how="left"
    )
    train = fix_train_annotation(train)
    train[train_col.annotation_length] = train[train_col.annotation].apply(len)
    train = add_validation_id(train)
    return train


def load_test_df(
    test_path: str, features_path: str, pt_notes_path: str
) -> pd.DataFrame:
    """テスト用データフレームを読み込む

    Args:
        test_path (str): テスト用データCSVのパス
        features_path (str): コンテキスト文章のCSVのパス
        pt_notes_path (str): 患者ノート文章のCSVのパス

    Returns:
        pd.DataFrame: テスト用データフレーム
    """
    test = pd.read_csv(test_path)
    features = pd.read_csv(features_path)
    features = preprocess_features(features)
    patient_notes = pd.read_csv(pt_notes_path)
    test = test.merge(
        features, on=[train_col.feature_num, train_col.case_num], how="left"
    )
    test = test.merge(
        patient_notes, on=[train_col.pn_num, train_col.case_num], how="left"
    )
    test[train_col.clean_text] = test[train_col.pn_history].apply(clean_spaces)
    return test


def load_test_df_fast(
    test_path: str, features_path: str, pt_notes_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """高速化テスト用データフレームを読み込む

    Args:
        test_path (str): テスト用データCSVのパス
        features_path (str): コンテキスト文章のCSVのパス
        pt_notes_path (str): 患者ノート文章のCSVのパス

    Returns:
        pd.DataFrame: テスト用ソート済みデータフレーム
        pd.DataFrame: テスト用データフレーム
        np.ndarray: ソート用インデックス行列
    """
    test = load_test_df(test_path, features_path, pt_notes_path)
    input_lengths: List[int] = []
    progress = tqdm(
        zip(
            test[train_col.pn_history].fillna("").values,
            test[train_col.feature_text].fillna("").values,
        ),
        total=len(test),
    )
    for text, feature_text in progress:
        length = len(
            Config.model.tokenizer(test, feature_text, add_special_tokens=True)[
                "input_ids"
            ]
        )
        input_lengths.append(length)
    test[train_col.input_lengths] = input_lengths
    length_sorted_idx = np.argsort([-1 * _len for _len in input_lengths])
    sort_test = test.iloc[length_sorted_idx]
    sorted_input_length = sort_test[train_col.input_lengths].values
    batch_max_length = np.zeros_like(sorted_input_length)
    bs = Config.batch_size
    for i in range((len(sorted_input_length) // bs) + 1):
        batch_max_length[i * bs : (i + 1) * bs] = np.max(
            sorted_input_length[i * bs : (i + 1) * bs]
        )
    sort_test["batch_max_length"] = batch_max_length
    return test, sort_test, length_sorted_idx


class BaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.feature_texts = df[train_col.feature_text].values
        self.pn_history = df[train_col.pn_history].values
        self.tokenizer = Config.model.tokenizer
        self.max_length = Config.model.max_length

    def __len__(self) -> int:
        return len(self.feature_texts)

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def prepare_input(
        self,
        text: str,
        feature_text: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """文章とコンテキストをtoken化する

        Args:
            text (str): 文章
            feature_text (str): コンテキスト

        Returns:
            Dict[str, torch.Tensor]: token化した文章
        """
        max_length = self.max_length if max_length is None else max_length
        inputs = self.tokenizer(
            text=text,
            text_pair=feature_text,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=False,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

    def create_label(
        self, text: str, annotation_length: int, location_list: List[str]
    ) -> torch.Tensor:
        """labelを作成する

        Args:
            text (str): 文章
            annotation_length (int): アノテーションの長さ
            location_list (List[str]): アノテーションのリスト

        Returns:
            torch.Tensor: 2値ラベル行列
        """
        encoded = self.tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offset_mapping = encoded["offset_mapping"]
        ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
        label = np.zeros(len(offset_mapping))
        label[ignore_idxes] = -1
        if annotation_length != 0:
            for location in location_list:
                for loc in [s.split() for s in location.split(";")]:
                    start_idx = -1
                    end_idx = -1
                    start, end = int(loc[0]), int(loc[1])
                    for idx in range(len(offset_mapping)):
                        if (start_idx == -1) & (start < offset_mapping[idx][0]):
                            start_idx = idx - 1
                        if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                            end_idx = idx + 1
                    if start_idx == -1:
                        start_idx = end_idx
                    if (start_idx != -1) & (end_idx != -1):
                        label[start_idx:end_idx] = 1
        return torch.tensor(label, dtype=torch.float)


class TrainDataset(BaseDataset):
    """訓練用データセット"""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.annotation_length = df[train_col.annotation_length].values
        self.location = df[train_col.location].values

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs = self.prepare_input(self.pn_history[index], self.feature_texts[index])
        labels = self.create_label(
            self.pn_history[index], self.annotation_length[index], self.location[index]
        )
        return inputs, labels


class TestDataset(BaseDataset):
    """予測用データセット"""

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        inputs = self.prepare_input(self.pn_history[index], self.feature_texts[index])
        return inputs


class TestDatasetFast(BaseDataset):
    """高速化した予測用データセット"""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.batch_max_len = df[train_col.batch_max_length].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        inputs = self.prepare_input(
            self.pn_history[index], self.feature_texts[index], self.batch_max_len[index]
        )
        return inputs
