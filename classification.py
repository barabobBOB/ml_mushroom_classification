from typing import Tuple

import cuml
import numpy as np
import pandas as pd
import sklearn.model_selection

RANDOM_SEED = 42


def load_data_from_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    csv 파일로부터 X, Y 데이터를 가져옵니다.

    Args:
        csv_path (str): csv 파일의 경로입니다.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: array 형태의 X, Y 값입니다.
    """

    label_encoder = sklearn.preprocessing.LabelEncoder()

    data: pd.DataFrame = pd.read_csv(csv_path)
    data.drop("veil-type", axis=1, inplace=True)  # 종류가 하나밖에 없음.
    x: pd.DataFrame = pd.get_dummies(data.drop("class", axis=1))
    x = x.astype(dtype="float32")
    y: pd.Series = label_encoder.fit_transform(data["class"])
    y = y.astype(dtype="float32")

    return x, y


def split_data(
    x: pd.DataFrame, y: pd.Series, test_ratio: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    데이터를 train, test 데이터로 나눕니다.

    Args:
        x (pd.DataFrame): 데이터
        y (pd.Series): 레이블
        test_ratio (float): 테스트셋의 비율(0~1)
        random_state (int): 셔플에 사용할 랜덤 시드

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            train_x, test_x, train_y, test_y 입니다.
    """
    return sklearn.model_selection.train_test_split(
        x, y, test_size=test_ratio, random_state=random_state, shuffle=True
    )
