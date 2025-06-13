import streamlit as st
import pandas as pd

# (gunakan kode sebelumnya untuk menghasilkan df_result)
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_original = X.copy()

X_discrete = X.copy()
for col in X.columns:
    kmeans = KMeans(n_clusters=3, random_state=42)
    X_discrete[col] = kmeans.fit_predict(X[[col]])

# Split data
X_train_o, X_test_o, y_train, y_test = train_test_split(X_original, y, test_size=0.3, random_state=1)
X_train_d, X_test_d, _, _ = train_test_split(X_discrete, y, test_size=0.3, random_state=1)

# Model training
nb_original = GaussianNB().fit(X_train_o, y_train)
dt_original = DecisionTreeClassifier().fit(X_train_o, y_train)

nb_discrete = GaussianNB().fit(X_train_d, y_train)
dt_discrete = DecisionTreeClassifier().fit(X_train_d, y_train)

result = {
    'Model': ['Naive Bayes', 'Naive Bayes', 'Decision Tree', 'Decision Tree'],
    'Data': ['Original', 'Diskritisasi', 'Original', 'Diskritisasi'],
    'Akurasi': [
        accuracy_score(y_test, nb_original.predict(X_test_o)),
        accuracy_score(y_test, nb_discrete.predict(X_test_d)),
        accuracy_score(y_test, dt_original.predict(X_test_o)),
        accuracy_score(y_test, dt_discrete.predict(X_test_d))
    ]
}
df_result = pd.DataFrame(result)
print(df_result)

import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(data=df_result, x='Model', y='Akurasi', hue='Data')
plt.title("Perbandingan Akurasi Klasifikasi")
plt.ylim(0.8, 1.0)
plt.show()

st.title("Perbandingan Klasifikasi Dataset Iris")
st.write("Hasil klasifikasi sebelum dan sesudah diskritisasi dengan K-Means.")

st.dataframe(df_result)