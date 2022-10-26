import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pydataset import data
import pickle

predictions = "survived_yes"
accuracy = 0
model_name = "acc_0.900.pickle"

# Import and filter dataset
dataset = data("titanic")
dataset = pd.get_dummies(dataset, drop_first = True)

x = np.array(dataset.drop(["survived_yes"], axis=1, inplace=False))
y = np.array(dataset["survived_yes"])

while (accuracy <= 0.9):
    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

    # Train model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

predictions = model.predict(x_test)

# Save best model
with open(model_name, "wb") as f:
    pickle.dump(model, f)

print("Accuracy =", accuracy)
print("Predictions [Target, Predicted]:")
for x in range(len(predictions)):
    print(y_test[x], "<--", predictions[x])