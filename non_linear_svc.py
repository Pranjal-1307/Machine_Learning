#NON-LINEAR SVC

#USING ARRAY
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample data: [Age, Salary, Purchased]
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
    [35, 65000, 1],
    [26, 80000, 1],
    [29, 43000, 0],
    [32, 135000, 1],
    [30, 87000, 1],
    [26, 32000, 0],
    [27, 17000, 0]
])

# Split features and labels
X = data[:, :2]   # Age and Salary
y = data[:, 2]    # Purchased

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train with polynomial kernel
clf = SVC(kernel='poly', degree=4)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot decision boundary
x_min, x_max = X_test[:, 0].min()-1, X_test[:, 0].max()+1
y_min, y_max = X_test[:, 1].min()-1, X_test[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.Paired)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.title("SVM with Polynomial Kernel (Degree=4)")
plt.xlabel("Age (scaled)")
plt.ylabel("Salary (scaled)")
plt.show()

#USING CSV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and prepare data
data = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:, [2, 3]]
y = data.iloc[:, 4]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train with polynomial kernel
clf = SVC(kernel='poly', degree=4)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot decision boundary
x_min, x_max = X_test[:, 0].min()-1, X_test[:, 0].max()+1
y_min, y_max = X_test[:, 1].min()-1, X_test[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.Paired)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.title("SVM with Polynomial Kernel (Degree=4)")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()