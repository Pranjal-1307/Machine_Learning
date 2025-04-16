#LINEAR SVC

#USING ARRAY
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample data: [Age, EstimatedSalary, Purchased]
data = np.array([
    [19, 19000, 0],
    [35, 20000, 0],
    [26, 43000, 0],
    [27, 57000, 0],
    [19, 76000, 0],
    [27, 58000, 0],
    [27, 84000, 1],
    [32, 150000, 1],
    [25, 33000, 0],
    [35, 65000, 1]
])

# Split features and labels
X = data[:, :2]   # Age, Salary
y = data[:, 2]    # Purchased

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
clf = SVC(kernel='linear', random_state=0)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X_test[:, 0].min(), X_test[:, 0].max())
yy = a * xx - (clf.intercept_[0] / w[1])
plt.plot(xx, yy)
plt.axis("off")
plt.show()


#USING CSV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("Social_Network_Ads.csv")
print(data)
X = data.iloc[:, [2, 3]]  # Age and EstimatedSalary
y = data.iloc[:, 4]       # Purchased
print(x) #x.head()
print(y) #y.head()


# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and predict
clf = SVC(kernel='linear', random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X_test[:, 0].min(), X_test[:, 0].max())
yy = a * xx - clf.intercept_[0] / w[1]
plt.plot(xx, yy)
plt.axis("off")
plt.show()
