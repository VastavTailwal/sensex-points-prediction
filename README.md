# Sensex Points Prediction Using LSTM

This project uses Long Short-Term Memory (LSTM) networks to predict Sensex points for the next 30 days based on historical data. The model processes key features affecting Sensex points and provides forecasts for future values.

## Project Overview

- **Objective**: To predict Sensex points for the next 30 days using an LSTM-based model for multivariate time series forecasting.
- **Approach**: 
  - Data processing, feature selection, and scaling techniques were applied.
  - An LSTM model was developed and trained on the processed dataset.
  - A web app was built to provide easy predictions for users.

## Steps Involved

1. **Data Processing**:
   - Identified and processed the key features influencing Sensex points.
   - Consolidated and cleaned the dataset for use in training the model.

2. **Data Preprocessing**:
   - Selected relevant features that contribute to accurate predictions.
   - Applied scaling techniques to normalize the data.

3. **LSTM Model Development**:
   - Built and trained a multivariate LSTM model on the processed data.
   - The model's performance was evaluated using Mean Squared Error (MSE).

4. **Web App Development**:
   - Developed a simple web application that allows users to predict Sensex points for the next 30 days.

## Model Performance

- **Train MSE**: 0.0011
- **Test MSE**: 0.0316

The model achieved a low error on the training data, while the test MSE indicates good generalization to unseen data.

## Usage Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Sensex-points-prediction.git
    cd "Sensex Points Prediction"
    ```

2. Install dependencies:
    ```bash
    pip install -e .
    ```

3. Run the web application:
    ```bash
    python app.py
    ```

4. Open the app in your browser, and click on the "Predict" button to forecast Sensex points for the next 30 days.

## Technologies Used

- Python
- TensorFlow/Keras (for LSTM model)
- Streamlit (for the web app)
- Pandas (for data processing)
- Scikit-learn (for scaling and preprocessing)

## Contributing

Feel free to fork this repository and contribute improvements, bug fixes, or new features.
