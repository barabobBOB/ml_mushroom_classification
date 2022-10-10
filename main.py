import classification
import matplotlib.pyplot as plt

x, y = classification.load_data_from_csv("mushrooms.csv")

train_x, test_x, train_y, test_y = classification.split_data(
    x, y, 0.2, classification.RANDOM_SEED
)

modeling_score = {}

for name in classification.classifier_names:
    classifier = classification.classifiers[name]
    classifier.fit(train_x, train_y)
    modeling_score[name] = classifier.score(test_x, test_y)*100
    print(f"{name} score: {classifier.score(test_x, test_y)}")

score_frame: classification.pd.Series = classification.pd.Series(modeling_score, name="score")
score_frame.plot.bar(color=['r', 'c', 'g', 'b'])
plt.show()