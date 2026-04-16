# 3D Point Cloud Manufacturing Feasibility App

This repository contains a Streamlit app for the ISE 5334 point-cloud assignment. The app is designed to support the Question 4 workflow by building a broader computation pipeline system for manufacturing feasibility classification from 3D point cloud `.ply` files.

 \\\\

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

 



## Question 5

For Question 5, the GitHub repository contains the Python code, documentation, and package requirements needed to reproduce the computation pipeline system. The Streamlit app turns the Question 4 analysis into an interactive online service by allowing users to upload labeled point-cloud datasets, compare 10 machine-learning pipelines, evaluate F1 scores, inspect misclassified samples, and score new designs.

 


