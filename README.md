# Housing_Pricing
House price prediction on the Ames, Iowa dataset (Kaggle): data cleaning, feature engineering, regularized linear and boosting models, and an ensemble evaluated with 5-fold cross-validation on the log price.
# House Price Prediction (Ames, Iowa)

Supervised regression project: predict the sale price of a house from its
characteristics (size, quality, age, neighborhood). Based on the Kaggle
competition **House Prices: Advanced Regression Techniques**.

## Goal and metric

- Target: `SalePrice` (USD), modeled as `log1p(SalePrice)` because the raw price is right-skewed (skewness 1.88 vs 0.12 after the log).
- Main metric: RMSE on the log price (the Kaggle metric), with R² and MAE in dollars as readable extras.

## Dataset

| File | Content |
|---|---|
| `train.csv` | 1,460 houses, 79 features, `Id` and `SalePrice` |
| `test.csv` | 1,459 houses, same columns without `SalePrice` |
| `data_description.txt` | Meaning of every column and category |

Data is downloaded from Kaggle and is not included in this repository.

## Workflow

1. Problem definition and metric
2. Loading and inspection (shapes, types, missing values)
3. Target analysis and log transform
4. Feature exploration (correlations, boxplots)
5. Outlier removal (2 houses with `GrLivArea` > 4,000 sold at low prices)
6. Missing values: `"None"` or 0 where NA means "no such feature", neighborhood median for `LotFrontage`, mode or median otherwise (statistics from train only)
7. Feature engineering: `TotalSF`, `TotalBathrooms`, `HouseAge`, `YearsSinceRemodel`, `HasPool`, `HasGarage`, `HasFireplace`, ordered quality scales, log of skewed columns, one-hot encoding
8. Model comparison with the same 5-fold cross-validation
9. Tuning (grid search) and ensemble
10. Diagnostics: predicted vs actual, residuals, feature importance
11. Interpretation
12. Prediction on the test set and `submission.csv`

## Results

| Model | CV RMSE (log price) |
|---|---|
| Baseline (mean) | XX |
| Ridge | XX |
| Lasso | XX |
| ElasticNet | XX |
| Random Forest | XX |
| Gradient Boosting | XX |
| XGBoost | XX |
| LightGBM | XX |
| **Ensemble (best 3)** | **XX** |

Final model: R² = XX, MAE = XX USD (out-of-fold predictions).

Main price drivers: XX (for example overall quality, living area, neighborhood).

## Good practices used

- Imputers and scalers fitted on training folds only (scikit-learn pipelines) to avoid data leakage
- Same folds for every model so the comparison is fair
- Train and test encoded together so both have identical columns
- Score judged on cross-validation, never on the training set

## Repository structure

```
.
├── notebook.ipynb        # full analysis, runs top to bottom
├── submission.csv        # Id, SalePrice (1,459 rows)
├── model.joblib          # saved final model
├── figures/              # saved charts
├── report/               # report and slides
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the Kaggle competition page, place them
next to the notebook, then run `notebook.ipynb` from top to bottom.

## Limits

The model is trained on Ames sales from 2006 to 2010. It should not be used to
price houses in another city or period.

## Authors

Your name (and team members)
