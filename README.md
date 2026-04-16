# 3D Point Cloud Manufacturing Feasibility App

This repository contains a Streamlit app for the ISE 5334 point-cloud assignment. The app is designed to support the Question 4 workflow by building a broader computation pipeline system for manufacturing feasibility classification from 3D point cloud `.ply` files.

## What the app does

The app lets a user:

- upload a **zip file** containing two folders named `feasible/` and `infeasible/`
- read `.ply` point-cloud files
- extract geometric and density-based features
- augment the feature set with unsupervised-learning features
- evaluate **10 machine-learning pipelines**
- compare **test-set F1 scores** against the earlier exam benchmark
- identify and inspect **misclassified samples**
- upload a new `.ply` file and score it with the best model

## Pipeline summary

The app implements the following 10 pipelines:

1. Logistic Regression + StandardScaler
2. Logistic Regression + RobustScaler
3. Logistic Regression + PCA
4. Random Forest (balanced)
5. Random Forest shallow (balanced)
6. Extra Trees (balanced)
7. Gradient Boosting
8. SVM (RBF)
9. KNN
10. MLP neural network

## Feature engineering included

The app extracts a richer feature set than the earlier exam baseline. Features include:

- number of points
- centroids
- minima and maxima
- ranges
- standard deviations and variances
- quartiles and medians
- k-nearest-neighbor distance summaries
- PCA eigenvalue-based shape descriptors
  - linearity
  - planarity
  - sphericity
  - omnivariance
  - anisotropy
  - eigenentropy
  - curvature

It also augments the model input with unsupervised-learning features:

- PCA-derived features
- K-means cluster label
- distances to K-means cluster centers

## Expected zip structure

Your uploaded dataset zip should contain folders named like this:

```text
my_dataset.zip
├── feasible/
│   ├── 001.stl_pointcloud.ply
│   ├── 002.stl_pointcloud.ply
│   └── ...
└── infeasible/
    ├── 101.stl_pointcloud.ply
    ├── 102.stl_pointcloud.ply
    └── ...
```

The app looks specifically for folder names containing `feasible` and `infeasible`.

## Files in this repo

- `app.py` — the full Streamlit app
- `requirements.txt` — Python dependencies
- `README.md` — setup and usage instructions

## Local setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Create and activate an environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

## How to upload this to GitHub

Since you already have a GitHub account, you can do the following.

### Option A: upload through the GitHub website

1. Create a new repository on GitHub.
2. Click **Add file** → **Upload files**.
3. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Commit the files.

### Option B: upload using git in Terminal

```bash
git init
git add .
git commit -m "Initial commit for point cloud Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

## How to deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Sign in with GitHub.
4. Click **New app**.
5. Choose your repository.
6. Set the main file path to:

```text
app.py
```

7. Click **Deploy**.

## Suggested writeup for Question 5

For Question 5, you can explain that the GitHub repository contains the Python code, documentation, and package requirements needed to reproduce the computation pipeline system. The Streamlit app turns the Question 4 analysis into an interactive online service by allowing users to upload labeled point-cloud datasets, compare 10 machine-learning pipelines, evaluate F1 scores, inspect misclassified samples, and score new designs.

You can also mention that the repository was structured so that another user can reproduce the workflow with minimal setup using `requirements.txt` and the included README instructions.

## Notes

- This app assumes **ASCII `.ply` files** with readable `x y z` coordinates.
- If your `.ply` files are in binary format, the parser in `app.py` would need to be extended.
- The earlier exam benchmark used approximately **0.75 test F1** as the best baseline, and the app compares new results against that benchmark.

