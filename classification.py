from typing import Tuple

import pandas as pd
import sklearn.model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB,
    CategoricalNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_SEED = 42

# 분류기 비교를 위하여 여러 분류기를 준비하였습니다.
classifiers = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED),
    "LogisticRegression": LogisticRegression(random_state=RANDOM_SEED),
    "SVC": SVC(random_state=RANDOM_SEED),
    "KNeighbors": KNeighborsClassifier(),
    "Categorial NaiveBayes": CategoricalNB(),
    "Multinomial NaiveBayes": MultinomialNB(),
    "Complement NaiveBayes": ComplementNB(),
    "Bernoulli NaiveBayes": BernoulliNB(),
    "Gaussian NaiveBayes": GaussianNB(),
    "Gaussian NaiveBayes param edit": GaussianNB(var_smoothing=1e-20),
}

classifier_names = list(classifiers.keys())


def load_data_from_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    csv 파일로부터 X, Y 데이터를 가져옵니다.

    Args:
        csv_path (str): csv 파일의 경로입니다.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: array 형태의 X, Y 값입니다.
    """

    label_encoder = sklearn.preprocessing.LabelEncoder()

    data: pd.DataFrame = pd.read_csv(csv_path, na_values="?")
    data.dropna(inplace=True)
    data.drop("veil-type", axis=1, inplace=True)  # 종류가 하나밖에 없음.
    x: pd.DataFrame = pd.get_dummies(data.drop("class", axis=1))
    x = x.astype(dtype="float32")
    print(x.shape)
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
