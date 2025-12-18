# Airbnb Pricing and Booking Analysis (Extra Credit Assignment)

This project analyzes Airbnb listings and calendar data from multiple U.S. cities to study **seasonality**, **pricing behavior**, and **booking probability**. Using InsideAirbnb data snapshots, we construct a **night-level panel dataset** and apply both **tree-based models (XGBoost)** and **neural networks** to predict nightly prices and booking outcomes.


## Dataset Description

The data is sourced from **InsideAirbnb** and includes two components for each city and snapshot:

- **Listings data**: static property characteristics (room type, capacity, reviews, host metrics, etc.)
- **Calendar data**: night-level availability and pricing information

Cities and snapshots analyzed:
- Austin (Dec 2024, Mar 2025)
- Chicago (Dec 2024, Mar 2025)
- Santa Cruz (Dec 2024, Mar 2025)
- Washington, DC (Dec 2024, Mar 2025)

Data is loaded directly from InsideAirbnb URLs and is https://data.insideairbnb.com/united-states with the given dates accordingly.
All the requirements libraries are already included in the code file

## Panel Data Construction

For each city and snapshot:
1. The most complete nightly price column (`adjusted_price`, `price`, or `base_price`) is automatically selected.
2. Calendar and listing datasets are merged at the **listing–date level**.
3. Prices are cleaned and converted to numeric format.
4. A binary booking indicator is created:
   - `is_booked = 1` if unavailable
   - `is_booked = 0` if available
5. Time-based features are extracted:
   - Month, day of week, week of year, day of year, weekend flag

To manage computational cost, each panel dataset is randomly sampled to **100,000 rows**, as recommended in the assignment.


## Seasonality Analysis

Seasonality patterns are explored using:
- Average price by month
- Booking probability by month
- Weekend vs weekday comparisons
- Monthly price trends by room type

## Modeling Approach

### Temporal Split
All models use a **time-aware split**:
- **Training**: January – September
- **Validation**: October – November
- **Testing**: December – February

This avoids information leakage across time.

### XGBoost Models

Two XGBoost models are trained for each dataset:
- **Regression**: Predict `price_numeric`
  - Metrics: RMSE, MAE
- **Classification**: Predict `is_booked`
  - Metrics: AUC, Accuracy

Categorical features (e.g., room type) are one-hot encoded, and missing numeric values are median-imputed.
plots are generated to interpret key drivers of pricing and booking behavior.

### Neural Network Models

For one representative dataset (Austin, March 2025), two feed-forward neural networks are trained:

- **Price Regression Network**
  - Loss: Mean Squared Error
  - Metrics: MAE
- **Booking Classification Network**
  - Loss: Binary Cross-Entropy
  - Metrics: Accuracy, AUC

All neural network training is logged using **TensorBoard** to visualize:
- Training vs validation loss
- MAE (regression)
- Accuracy and AUC (classification)

Screenshots of TensorBoard outputs are included in the `images/` directory.


## Results Summary

- Booking probability is generally easier to predict than exact nightly price.
- Weekend and seasonal effects significantly influence both price and booking likelihood.
- XGBoost models perform strongly with interpretable feature importance.
- Neural networks show stable learning behavior with limited overfitting, as observed through TensorBoard diagnostics.


## Repository Structure

