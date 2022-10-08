import cuml
from cuml.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import cudf
from typing import Tuple

RANDOM_SEED = 42

# 분류기 비교를 위하여 여러 분류기를 준비하였습니다.
classifiers = {
    "RandomForest": cuml.RandomForestClassifier(random_state=RANDOM_SEED),
    "LogisticRegression": cuml.LogisticRegression(random_state=RANDOM_SEED),
    "SVC": cuml.svm.SVC(random_state=RANDOM_SEED),
    "KNeighbors": cuml.neighbors.KNeighborsClassifier(),
    "NaiveBayes": cuml.naive_bayes.CategoricalNB(),
}

classifier_names = list(classifiers.keys())


def load_data_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    csv 파일로부터 X, Y 데이터를 가져옵니다.

    Args:
        csv_path (str): csv 파일의 경로입니다.

    Returns:
        Tuple[np.ndarray, np.ndarray]: array 형태의 X, Y 값입니다.
    """
    data: cudf.DataFrame = cudf.read_csv("mushrooms.csv")
    data.drop("veil-type", axis=1, inplace=True)  # 종류가 하나밖에 없음.

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()

    x: cudf.DataFrame = cudf.get_dummies(data.drop("class", axis=1))
    y: cudf.Series = label_encoder.fit_transform(data["class"])

    return x, y


def split_data(
    x: np.ndarray, y: np.ndarray, test_ratio: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터를 train, test 데이터로 나눕니다.

    Args:
        x (np.ndarray): 데이터
        y (np.ndarray): 레이블
        test_ratio (float): 테스트셋의 비율(0~1)
        random_state (int): 셔플에 사용할 랜덤 시드

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            train_x, test_x, train_y, test_y 입니다.
    """
    return cuml.train_test_split(
        x, y, test_size=test_ratio, random_state=random_state, shuffle=True
    )
