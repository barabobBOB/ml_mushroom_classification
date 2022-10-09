import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB

import classification

# 분류기 비교를 위하여 여러 분류기를 준비하였습니다.
classifiers = {
    "RandomForest": RandomForestClassifier(random_state=classification.RANDOM_SEED),
    "LogisticRegression": LogisticRegression(),
    "SVC": SVC(random_state=classification.RANDOM_SEED),
    "KNeighbors": KNeighborsClassifier(),
    "NaiveBayes": CategoricalNB(),
}

classifier_names = list(classifiers.keys())

x, y = classification.load_data_from_csv("mushrooms.csv")

train_x, test_x, train_y, test_y = classification.split_data(
    x, y, 0.2, classification.RANDOM_SEED
)

for name in classifier_names:
    classifier = classifiers[name]
    classifier.fit(train_x, train_y)
    print(f"{name} score: {classifier.score(test_x, test_y)}")
