import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

st.title("Machine Learning App")

st.write("""
# Select a dataset and classifier
Which one works best for you?
""")  # markdown format

#dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
st.write(f'Dataset = {dataset_name}')

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)  # check official sklearn docs why C is var name
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                    max_depth=params["max_depth"], random_state=1223)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

acc = accuracy_score(y_test, y_predict)
st.write(f'classifier = {classifier_name}')
st.write(f'ML accuracy = {acc}')

# Plot
st.write("""
# 2-D PCA Plot
""")
pca = PCA(2)
X_projected = pca.fit_transform(X)  # doesn't need y, it is unsupervised technique that doesn't need (column) labels
x1, x2 = X_projected[:, 0], X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")  # alpha is transparancy
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar()

#plt.show()
st.pyplot(fig)