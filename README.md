# House Price Prediction

Predicts the median house price of a California district using regression models trained on the California Housing dataset.
This was built as Task 3 of a machine learning internship.

## Dataset

- **Source:** California Housing dataset, loaded via `sklearn.datasets.fetch_california_housing(as_frame=True)`
- The dataset is downloaded automatically from scikit-learn on the first run. No CSV file is included.
- **Size:** 20,640 rows, 8 features
- **Target:** Median house value in units of $100,000 (e.g. a value of 2.5 means $250,000)
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Origin:** 1990 California census data, aggregated at census-block-group (district) level

## Models

Three regression models are trained and compared:
- Linear Regression
- Decision Tree Regressor (max_depth chosen by 5-fold CV on the training set)
- Random Forest Regressor (n_estimators=100)

## Results (test set, 20% holdout)

| Model                   | RMSE   | MAE    | R2     |
|-------------------------|--------|--------|--------|
| Linear Regression       | 0.7292 | 0.5261 | 0.5942 |
| Decision Tree (depth=8) | 0.6502 | 0.4483 | 0.6774 |
| Random Forest           | 0.5056 | 0.3285 | 0.8049 |

RMSE and MAE are in units of $100,000. Random Forest's RMSE of 0.5056 corresponds to roughly **$50,560** and its MAE of 0.3285 corresponds to roughly **$32,850**.
Cross-validation on the full dataset agrees with this ranking.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter notebook house_price_prediction.ipynb
   ```

   On the first run, scikit-learn will download the dataset automatically.
