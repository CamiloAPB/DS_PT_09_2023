from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

iris = load_iris()

X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
esc = StandardScaler()
X_esc = esc.fit_transform(X)

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_esc, y, train_size=0.8, random_state=42)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_flask = joblib.dump(knn, "knn_flask_esc.pkl")