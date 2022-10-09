import classification

x, y = classification.load_data_from_csv("mushrooms.csv")

train_x, test_x, train_y, test_y = classification.split_data(
    x, y, 0.2, classification.RANDOM_SEED
)

for name in classification.classifier_names:
    classifier = classification.classifiers[name]
    classifier.fit(train_x, train_y)
    print(f"{name} score: {classifier.score(test_x, test_y)}")
