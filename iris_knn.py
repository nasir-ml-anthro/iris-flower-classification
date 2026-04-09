import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# ১. ডেটা লোড করা
iris = load_iris()
X = iris.data  # ফিচার (ফুলের মাপ)
y = iris.target # টার্গেট (ফুলের জাত)

# ২. ডেটাকে ট্রেইনিং এবং টেস্টিং সেটে ভাগ করা (৮০% ট্রেইনিং, ২০% টেস্টিং)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ৩. মডেল তৈরি করা (KNN অ্যালগরিদম ব্যবহার করে)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ৪. প্রেডিকশন এবং একুরেসি চেক করা
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"মডেলের একুরেসি: {accuracy * 100}%")
