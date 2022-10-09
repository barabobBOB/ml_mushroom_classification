import cuml

import classification

# 분류기 비교를 위하여 여러 분류기를 준비하였습니다.
classifiers = {
    "RandomForest": cuml.RandomForestClassifier(
        random_state=classification.RANDOM_SEED
    ),
    "LogisticRegression": cuml.LogisticRegression(),
    "SVC": cuml.svm.SVC(random_state=classification.RANDOM_SEED),
    "KNeighbors": cuml.neighbors.KNeighborsClassifier(),
    "NaiveBayes": cuml.naive_bayes.CategoricalNB(),
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
