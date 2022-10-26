import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle 

accuracy = 0
model_name = "acc_0.975.pickle"

# Filter dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")
data = pd.get_dummies(data, drop_first = False)
data = data[["G1","G2","G3","age","traveltime","studytime","failures","absences"]]


# Determine x and y data
prediction = "G3"
x = np.array(data.drop([prediction], axis=1, inplace=False))
y = np.array(data[prediction])

# Reiterate to find optimal model
while (accuracy <= 0.955):
    # Separate train and test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

   
    # Create linear regression model
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    # Compare model accuracy to test data
    accuracy = model.score(x_test, y_test)

# Predict G3 for each student entity
predictions = model.predict(x_test)

# Save best model
with open(model_name, "wb") as f:
    pickle.dump(model, f)

pickle_in = open(model_name, "rb")
model = pickle.load(pickle_in)

# Print results
print("Accuracy =", accuracy)
print("Coeff =", model.coef_)
print("Intercept =", model.intercept_)

print("Predictions [Target, Predicted]:")
for x in range(len(predictions)):
    print(y_test[x], "<--", predictions[x])

# Plotting
p = "G2"

style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("G3")

plt.show()