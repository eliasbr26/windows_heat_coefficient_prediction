# Windows Heat Coefficient Prediction

A machine learning project for predicting thermal transmission coefficients (U-values) of windows using building energy performance data.

## Overview

This project uses Random Forest Regression to predict the heat transfer coefficient (`u_baie_vitree`) of windows based on various building characteristics and window properties. The model is trained on French DPE (Diagnostic de Performance Énergétique) data.

## Dataset

The project uses two main data sources:
- `Phebus_chauffage_allege_avec_new_R_new_S.xlsx`: Building heating data
- `batiment_groupe_dpe_representatif_logement.csv`: Representative building energy performance diagnostics

## Features

The model uses the following features to predict window U-values:

### Numerical Features
- `annee_construction_dpe`: Construction year
- `epaisseur_lame`: Thickness of the air gap between glass panes
- `vitrage_vir`: Low-emissivity coating indicator
- `surface_vitree_nord/sud/ouest/est`: Glazed surface area by orientation (North/South/West/East)
- `surface_habitable_logement`: Total habitable surface area
- `presence_balcon`: Presence of balcony

### Categorical Features
- `type_vitrage`: Type of glazing (simple, double, triple)
- `type_materiaux_menuiserie`: Frame material (wood, PVC, metal, etc.)
- `type_gaz_lame`: Gas fill type (air, argon, etc.)
- `type_fermeture`: Type of window closure/shutter

## Methodology

### Data Preprocessing
1. **Missing Value Handling**:
   - Numerical features: Imputed with mean values
   - Categorical features: Imputed with most frequent values

2. **Feature Engineering**:
   - One-hot encoding for categorical variables
   - Removal of rows where target variable (`u_baie_vitree`) is missing

3. **Train-Test Split**: 80/20 split with random state 42

### Model

- **Algorithm**: Random Forest Regressor
- **Parameters**: 
  - `n_estimators=30`
  - `random_state=42`

## Results

### Model Performance

```
Mean Absolute Error (MAE): 0.218
Mean Squared Error (MSE): 0.133
Root Mean Squared Error (RMSE): 0.365
R² Score: 0.885
```

The model achieves an **R² score of 0.885**, indicating that it explains approximately 88.5% of the variance in window U-values.

### Feature Importance

Top 10 most important features:

| Feature | Importance |
|---------|-----------|
| Simple glazing type | 48.56% |
| Low-E coating (VIR) | 14.58% |
| No window closure | 11.60% |
| Metal frame without thermal break | 5.25% |
| PVC frame | 3.05% |
| Air gap thickness | 2.92% |
| Habitable surface area | 2.44% |
| West-facing glazed area | 2.16% |
| East-facing glazed area | 2.13% |
| South-facing glazed area | 2.04% |

## Visualizations

The notebook includes:
1. **Actual vs Predicted Values Plot**: Shows model prediction accuracy
2. **Residuals Plot**: Displays prediction errors across different predicted values

## Requirements

```python
pandas
numpy
scikit-learn
matplotlib
openpyxl  # for Excel file reading
```

## Usage

```python
# Load and preprocess data
df = pd.read_csv("batiment_groupe_dpe_representatif_logement.csv")

# Train model
rf = RandomForestRegressor(n_estimators=30, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)
```

## Project Structure

```
windows_heat_coefficient_prediction/
├── test.ipynb                                          # Main analysis notebook
├── batiment_groupe_dpe_representatif_logement.csv     # DPE dataset
├── Phebus_chauffage_allege_avec_new_R_new_S.xlsx     # Heating data
└── README.md                                          # This file
```

## Key Insights

1. **Glazing type** is by far the most important predictor (48.6% importance), with simple glazing having significantly higher U-values than double or triple glazing.

2. **Low-E coating** (VIR) is the second most important feature (14.6%), indicating substantial impact on thermal performance.

3. **Frame material** and **shutter presence** also play significant roles in determining window thermal efficiency.

4. The model shows good generalization with consistent residual distribution across predicted values.

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature selection to reduce dimensionality
- Testing alternative algorithms (XGBoost, LightGBM)
- Cross-validation for more robust performance estimates
- Analysis of prediction errors for different building types

## License

This project uses publicly available French DPE data for research purposes.

## Author

eliasbr26
