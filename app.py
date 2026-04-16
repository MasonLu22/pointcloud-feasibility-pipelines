import io
import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC

st.set_page_config(page_title="3D Point Cloud Pipeline App", layout="wide")


# -----------------------------
# Utilities
# -----------------------------
def read_ply(file_obj_or_path):
    """Read ASCII .ply with x, y, z columns. Returns n x 3 array."""
    if hasattr(file_obj_or_path, "read"):
        content = file_obj_or_path.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        else:
            text = content
        lines = text.splitlines()
    else:
        with open(file_obj_or_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

    try:
        header_end = lines.index("end_header")
    except ValueError:
        raise ValueError("This .ply file does not appear to contain a valid ASCII PLY header.")

    data_lines = lines[header_end + 1 :]
    pts = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue

    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        raise ValueError("No valid x, y, z coordinates were found in this .ply file.")
    return pts[:, :3]


def clean_points(points, zscore_thresh=4.0):
    """Simple outlier removal using coordinate-wise z scores."""
    if len(points) < 10:
        return points
    mu = points.mean(axis=0)
    sd = points.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    z = np.abs((points - mu) / sd)
    keep = (z < zscore_thresh).all(axis=1)
    cleaned = points[keep]
    if len(cleaned) < max(20, int(0.2 * len(points))):
        return points
    return cleaned


def knn_distance_stats(points, k=5):
    if len(points) < k + 1:
        return [np.nan, np.nan, np.nan]
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(points)
    dists, _ = nbrs.kneighbors(points)
    d = dists[:, 1:]
    row_mean = d.mean(axis=1)
    return [float(row_mean.mean()), float(row_mean.std()), float(row_mean.max())]


def pca_shape_features(points):
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 1e-12)
    l1, l2, l3 = eigvals
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    omnivariance = (l1 * l2 * l3) ** (1 / 3)
    anisotropy = (l1 - l3) / l1
    eigenentropy = -np.sum(eigvals * np.log(eigvals))
    curvature = l3 / (l1 + l2 + l3)
    return [float(v) for v in [l1, l2, l3, linearity, planarity, sphericity,
                               omnivariance, anisotropy, eigenentropy, curvature]]


def extract_features(points):
    pts = clean_points(points)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    centroid = pts.mean(axis=0)
    stds = pts.std(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    q25 = np.percentile(pts, 25, axis=0)
    q50 = np.percentile(pts, 50, axis=0)
    q75 = np.percentile(pts, 75, axis=0)
    knn_feats = knn_distance_stats(pts, k=5)
    shape_feats = pca_shape_features(pts)

    return {
        "num_points": int(len(pts)),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "std_x": float(stds[0]),
        "std_y": float(stds[1]),
        "std_z": float(stds[2]),
        "min_x": float(mins[0]),
        "min_y": float(mins[1]),
        "min_z": float(mins[2]),
        "max_x": float(maxs[0]),
        "max_y": float(maxs[1]),
        "max_z": float(maxs[2]),
        "x_range": float(maxs[0] - mins[0]),
        "y_range": float(maxs[1] - mins[1]),
        "z_range": float(maxs[2] - mins[2]),
        "q25_x": float(q25[0]),
        "q25_y": float(q25[1]),
        "q25_z": float(q25[2]),
        "median_x": float(q50[0]),
        "median_y": float(q50[1]),
        "median_z": float(q50[2]),
        "q75_x": float(q75[0]),
        "q75_y": float(q75[1]),
        "q75_z": float(q75[2]),
        "var_x": float(np.var(x)),
        "var_y": float(np.var(y)),
        "var_z": float(np.var(z)),
        "knn_mean": knn_feats[0],
        "knn_sd": knn_feats[1],
        "knn_max": knn_feats[2],
        "eig1": shape_feats[0],
        "eig2": shape_feats[1],
        "eig3": shape_feats[2],
        "linearity": shape_feats[3],
        "planarity": shape_feats[4],
        "sphericity": shape_feats[5],
        "omnivariance": shape_feats[6],
        "anisotropy": shape_feats[7],
        "eigenentropy": shape_feats[8],
        "curvature": shape_feats[9],
    }


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def make_pipelines(random_state=42):
    return {
        "LR_standard": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=random_state))
        ]),
        "LR_robust": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=random_state))
        ]),
        "LR_pca": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, random_state=random_state)),
            ("model", LogisticRegression(max_iter=5000, random_state=random_state))
        ]),
        "RF_basic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, random_state=random_state, class_weight="balanced"
            ))
        ]),
        "RF_shallow": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=3,
                random_state=random_state, class_weight="balanced"
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=400, random_state=random_state, class_weight="balanced"
            ))
        ]),
        "GBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=random_state))
        ]),
        "SVM_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=2000, random_state=random_state))
        ]),
    }


@st.cache_data(show_spinner=False)
def build_dataset_from_zip(zip_bytes):
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        zippath = os.path.join(tmpdir, "dataset.zip")
        with open(zippath, "wb") as f:
            f.write(zip_bytes)

        with zipfile.ZipFile(zippath, "r") as zf:
            zf.extractall(tmpdir)

        root = Path(tmpdir)
        ply_files = list(root.rglob("*.ply"))
        if not ply_files:
            raise ValueError("No .ply files were found in the uploaded zip file.")

        for file_path in ply_files:
            parts_lower = [p.lower() for p in file_path.parts]
            if "feasible" in parts_lower:
                label = 1
            elif "infeasible" in parts_lower:
                label = 0
            else:
                continue

            pts = read_ply(str(file_path))
            feats = extract_features(pts)
            feats["label"] = label
            feats["file"] = file_path.name
            rows.append(feats)

    if not rows:
        raise ValueError(
            "The zip file was read, but no labeled files were found. Make sure it contains folders named feasible/ and infeasible/."
        )

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ["label", "file"]]
    X_base = df[feature_cols].copy()
    X_tmp = X_base.fillna(X_base.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tmp)

    pca_for_features = PCA(n_components=min(5, X_scaled.shape[1]), random_state=1)
    pca_scores = pca_for_features.fit_transform(X_scaled)
    for i in range(pca_scores.shape[1]):
        X_base[f"pca_feat_{i+1}"] = pca_scores[:, i]

    kmeans = KMeans(n_clusters=4, random_state=1, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    X_base["cluster_label"] = cluster_labels

    for j in range(4):
        X_base[f"dist_cluster_{j}"] = np.linalg.norm(X_scaled - kmeans.cluster_centers_[j], axis=1)

    final_df = X_base.copy()
    final_df["label"] = df["label"].values
    final_df["file"] = df["file"].values
    return final_df


def evaluate_pipelines(df, test_size=0.2, random_state=42, n_splits=5):
    X = df.drop(columns=["label", "file"])
    y = df["label"]
    files = df["file"]

    X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
        X, y, files, test_size=test_size, random_state=random_state, stratify=y
    )

    pipelines = make_pipelines(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_rows = []
    test_rows = []
    fitted_models = {}

    for name, pipe in pipelines.items():
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            Xtr = X_train.iloc[tr_idx]
            Xva = X_train.iloc[va_idx]
            ytr = y_train.iloc[tr_idx]
            yva = y_train.iloc[va_idx]

            model = clone(pipe)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            fold_scores.append(f1_score(yva, pred))

        cv_rows.append({
            "Pipeline": name,
            "CV_F1_Mean": np.mean(fold_scores),
            "CV_F1_SD": np.std(fold_scores),
        })

        final_model = clone(pipe)
        final_model.fit(X_train, y_train)
        pred_test = final_model.predict(X_test)

        fitted_models[name] = final_model
        test_rows.append({
            "Pipeline": name,
            "Test_Accuracy": accuracy_score(y_test, pred_test),
            "Test_Precision": precision_score(y_test, pred_test, zero_division=0),
            "Test_Recall": recall_score(y_test, pred_test, zero_division=0),
            "Test_F1": f1_score(y_test, pred_test, zero_division=0),
        })

    cv_results = pd.DataFrame(cv_rows).sort_values("CV_F1_Mean", ascending=False)
    test_results = pd.DataFrame(test_rows).sort_values("Test_F1", ascending=False)
    best_name = test_results.iloc[0]["Pipeline"]
    best_model = fitted_models[best_name]
    best_pred = best_model.predict(X_test)

    mis_mask = (best_pred != y_test.values)
    misclassified_df = pd.DataFrame({
        "file": file_test.values,
        "true_label": y_test.values,
        "pred_label": best_pred,
    }).loc[mis_mask].copy()

    diagnosis = pd.DataFrame({
        "Misclassified_Mean": X_test.loc[mis_mask].mean(numeric_only=True),
        "CorrectlyClassified_Mean": X_test.loc[~mis_mask].mean(numeric_only=True),
    })
    diagnosis["Abs_Diff"] = (diagnosis["Misclassified_Mean"] - diagnosis["CorrectlyClassified_Mean"]).abs()
    diagnosis = diagnosis.sort_values("Abs_Diff", ascending=False)

    return {
        "cv_results": cv_results,
        "test_results": test_results,
        "best_name": best_name,
        "best_model": best_model,
        "best_pred": best_pred,
        "X_test": X_test,
        "y_test": y_test,
        "file_test": file_test,
        "misclassified_df": misclassified_df,
        "diagnosis": diagnosis,
        "classification_report": classification_report(y_test, best_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_test, best_pred),
    }


def compare_with_exam_baseline(test_results):
    baseline = 0.75  # from the earlier exam writeup
    out = test_results.copy()
    out["Exam_Best_F1"] = baseline
    out["Improvement_vs_Exam"] = out["Test_F1"] - baseline
    return out.sort_values("Test_F1", ascending=False)


def plot_point_cloud(points, title="3D Point Cloud"):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    n = len(points)
    if n > 8000:
        idx = np.random.choice(n, 8000, replace=False)
        pts = points[idx]
    else:
        pts = points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig


# -----------------------------
# App UI
# -----------------------------
st.title("3D Point Cloud Manufacturing Feasibility App")
st.markdown(
    "Upload a zip file that contains two folders named **feasible/** and **infeasible/**. "
    "The app will extract point-cloud features, run 10 pipelines, compare F1 scores, "
    "diagnose misclassified samples, and let you score a new `.ply` file with the best model."
)

with st.sidebar:
    st.header("Settings")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=1, max_value=9999, value=42, step=1)
    n_splits = st.slider("CV folds", 3, 10, 5, 1)

zip_file = st.file_uploader("Upload labeled dataset zip", type=["zip"])

if zip_file is not None:
    try:
        df = build_dataset_from_zip(zip_file.getvalue())
        st.success(f"Dataset loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Samples", df.shape[0])
        with c2:
            st.metric("Feasible", int((df['label'] == 1).sum()))
        with c3:
            st.metric("Infeasible", int((df['label'] == 0).sum()))

        st.subheader("Preview of extracted feature table")
        st.dataframe(df.head())

        if st.button("Run pipeline comparison"):
            with st.spinner("Running feature engineering and evaluating 10 pipelines..."):
                results = evaluate_pipelines(df, test_size=test_size, random_state=int(random_state), n_splits=n_splits)
                st.session_state["results"] = results
                st.session_state["df"] = df

    except Exception as e:
        st.error(f"Error while reading dataset: {e}")

if "results" in st.session_state:
    results = st.session_state["results"]
    df = st.session_state["df"]

    st.subheader("Cross-validated training comparison")
    st.dataframe(results["cv_results"], use_container_width=True)

    st.subheader("Held-out test results")
    test_results = results["test_results"].copy()
    st.dataframe(test_results, use_container_width=True)

    st.subheader("Comparison with exam baseline")
    comp = compare_with_exam_baseline(test_results)
    st.dataframe(comp, use_container_width=True)

    st.markdown(
        f"**Best pipeline:** `{results['best_name']}`  \\\n"
        f"**Best test F1:** `{test_results.iloc[0]['Test_F1']:.3f}`  \\\n"
        f"**Improvement vs exam benchmark (0.75):** `{test_results.iloc[0]['Test_F1'] - 0.75:.3f}`"
    )

    st.subheader("Test-set F1 chart")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(test_results["Pipeline"], test_results["Test_F1"])
    ax.axhline(0.75, linestyle="--", linewidth=1)
    ax.set_ylabel("Test F1")
    ax.set_xticklabels(test_results["Pipeline"], rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Classification report for best pipeline")
    st.text(results["classification_report"])

    st.subheader("Confusion matrix")
    st.write(results["confusion_matrix"])

    st.subheader("Misclassified samples")
    if results["misclassified_df"].empty:
        st.info("No misclassified samples in the current test split.")
    else:
        st.dataframe(results["misclassified_df"], use_container_width=True)

    st.subheader("Top feature differences for diagnosis")
    st.dataframe(results["diagnosis"].head(15), use_container_width=True)

    st.subheader("Score a new point cloud with the best model")
    new_ply = st.file_uploader("Upload a single .ply file for prediction", type=["ply"], key="single_ply")
    if new_ply is not None:
        try:
            pts = read_ply(io.BytesIO(new_ply.getvalue()))
            fig3d = plot_point_cloud(pts, title=new_ply.name)
            st.pyplot(fig3d)

            feats = extract_features(pts)
            row = pd.DataFrame([feats])

            # add same unsupervised-style features using current dataset columns if needed
            train_df = df.drop(columns=["label", "file"])
            train_tmp = train_df.fillna(train_df.median(numeric_only=True))
            scaler = StandardScaler()
            scaler.fit(train_tmp)
            row_tmp = row.fillna(train_tmp.median(numeric_only=True))
            row_scaled = scaler.transform(row_tmp)
            train_scaled = scaler.transform(train_tmp)

            pca_for_features = PCA(n_components=min(5, train_scaled.shape[1]), random_state=1)
            pca_for_features.fit(train_scaled)
            pca_scores = pca_for_features.transform(row_scaled)
            for i in range(pca_scores.shape[1]):
                row[f"pca_feat_{i+1}"] = pca_scores[:, i]

            kmeans = KMeans(n_clusters=4, random_state=1, n_init=20)
            kmeans.fit(train_scaled)
            cluster_label = kmeans.predict(row_scaled)[0]
            row["cluster_label"] = cluster_label
            for j in range(4):
                row[f"dist_cluster_{j}"] = np.linalg.norm(row_scaled - kmeans.cluster_centers_[j], axis=1)

            # align columns
            Xcols = df.drop(columns=["label", "file"]).columns
            row = row.reindex(columns=Xcols)

            best_model = results["best_model"]
            pred = int(best_model.predict(row)[0])
            label = "Feasible" if pred == 1 else "Infeasible"
            st.success(f"Predicted class: {label}")
            st.dataframe(row, use_container_width=True)
        except Exception as e:
            st.error(f"Could not score the uploaded .ply file: {e}")

st.markdown("---")
st.caption("Built for the ISE 5334 point-cloud pipeline assignment.")
