# Airfoil Self-Noise Surrogate Model (XGBoost)

## Executive Summary
This repository contains a machine learning pipeline designed to predict airfoil Scaled Sound Pressure Level (SSPL) without relying on traditional energy-expensive wind tunnel testing or numerical solvers. 

By training an **XGBoost regression model** on NASA wind-tunnel data, this tool maps complex aerodynamic parameters directly to acoustic signatures. The model achieves a 95.83% R-squared accuracy (1.33 MSE) and is optimized for rapid, solver-free prototyping by strategically eliminating high-cost physical variables in prototyping phase.

---

## The Engineering Problem
In aviation, accurately predicting noise levels based on operational parameters (frequency, angle of attack,suction side displacement thickness, chord length, free stream velocity) traditionally requires heavy fluid dynamics simulations. 

While data-driven methods bypass these bottlenecks, they often rely on variables that must first be calculated using complex numerical methods (such as *suction side displacement thickness*). Including these variables in a predictive model defeats the purpose of rapid machine learning analysis. 

## The Solution & Architecture
This project builds a highly accurate surrogate model while actively stripping out computationally expensive dependencies.

* **Algorithm:** XGBoost Regressor
* **Dataset:** NASA Airfoil Self-Noise Dataset (>1,500 wind tunnel measurements)
* **Feature Engineering:** Conducted feature correlation analysis using Pearson and Spearman correlation (Heatmap generation) to evaluate feature importance.
* **Optimization:** Strategically dropped the delta (suction side displacement thickness) parameter. While this slightly impacts theoretical max accuracy, it transforms the model into a viable, real-world engineering tool that does not require prior numerical solver runs.

## Results & Performance
The model accurately captures the highly non-linear aerodynamics associated with acoustic emissions. 
* **R-squared Score:** 95.83%
* **Mean Squared Error (MSE):** 1.33

## Tech Stack & Dependencies
* **Python 3.13.12**
* `pandas` & `numpy` (Data Processing)
* `scikit-learn` (Validation & Preprocessing)
* `xgboost` (Core Regression Model)
* `matplotlib` & `seaborn` (Visualization)

## Dependencies & Usage 
To run this surrogate model, download the repository and ensure your local Python environment has the following libraries installed:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
