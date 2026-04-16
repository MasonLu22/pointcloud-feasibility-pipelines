
import streamlit as st
import zipfile
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

st.title("3D Point Cloud Feasibility Classification")

# =========================================================
# Q1: Feature Extraction from 3D Point Clouds
# =========================================================

def read_ply(file_path):
    """
    Read ASCII .ply file and return x,y,z points
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_end = lines.index("end_header\n")
    pts = np.loadtxt(lines[header_end+1:])

    return pts[:, :3]


def extract_features(points):
    """
    Extract geometric summary features from point cloud
    """
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    features = {
        "num_points": len(points),
        "centroid_x": np.mean(x),
        "centroid_y": np.mean(y),
        "centroid_z": np.mean(z),
        "x_range": np.max(x) - np.min(x),
        "y_range": np.max(y) - np.min(y),
        "z_range": np.max(z) - np.min(z),
        "var_x": np.var(x),
        "var_y": np.var(y),
        "var_z": np.var(z),
    }

    return features


# =========================================================
# Q2: Dataset Construction and Evaluation Metrics
# =========================================================

def build_dataset(feasible_folder, infeasible_folder):

    rows = []

    for fn in os.listdir(feasible_folder):
        if fn.endswith(".ply"):
            pts = read_ply(os.path.join(feasible_folder, fn))
            feats = extract_features(pts)
            feats["label"] = 1
            feats["file"] = fn
            rows.append(feats)

    for fn in os.listdir(infeasible_folder):
        if fn.endswith(".ply"):
            pts = read_ply(os.path.join(infeasible_folder, fn))
            feats = extract_features(pts)
            feats["label"] = 0
            feats["file"] = fn
            rows.append(feats)

    df = pd.DataFrame(rows)

    return df


# =========================================================
# Q3: Feature Engineering Using Unsupervised Learning
# =========================================================

def augment_features(df):

    X = df.drop(columns=["label","file"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA features
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(X_scaled)

    df["pca1"] = pca_features[:,0]
    df["pca2"] = pca_features[:,1]
    df["pca3"] = pca_features[:,2]

    # clustering feature
    kmeans = KMeans(n_clusters=3, random_state=1)
    clusters = kmeans.fit_predict(X_scaled)

    df["cluster"] = clusters

    return df


# =========================================================
# Q4: Machine Learning Pipelines and Model Comparison
# =========================================================

def run_pipelines(X_train, X_test, y_train, y_test):

    pipelines = {

        "LogisticRegression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000))
            ]),

        "RandomForest":
            RandomForestClassifier(n_estimators=300),

        "GradientBoosting":
            GradientBoostingClassifier(),

        "ExtraTrees":
            ExtraTreesClassifier(n_estimators=300),

        "DecisionTree":
            DecisionTreeClassifier(),

        "SVM":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC())
            ]),

        "KNN":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ]),

        "MLP":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(max_iter=2000))
            ])
    }

    results = []

    best_model = None
    best_f1 = 0

    for name, model in pipelines.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        f1 = f1_score(y_test, pred)

        results.append({
            "Pipeline": name,
            "F1 Score": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)

    return results_df, best_model


# =========================================================
# Streamlit Interface (Interactive Application)
# =========================================================

uploaded_zip = st.file_uploader("Upload dataset zip (feasible / infeasible folders)", type="zip")

if uploaded_zip:

    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall("dataset")

    st.write("Dataset extracted.")

    feasible_folder = "dataset/feasible"
    infeasible_folder = "dataset/infeasible"

    df = build_dataset(feasible_folder, infeasible_folder)

    st.write("Dataset preview:")
    st.dataframe(df.head())

    df = augment_features(df)

    X = df.drop(columns=["label","file"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1
    )

    st.write("Running machine learning pipelines...")

    results, best_model = run_pipelines(X_train, X_test, y_train, y_test)

    st.write("Pipeline comparison (F1 score):")
    st.dataframe(results)

    st.write("Best model selected automatically.")
