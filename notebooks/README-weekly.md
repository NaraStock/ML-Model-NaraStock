# NaraStock-ML-Weekly

NaraStock is a forex currency price movement prediction project (EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD) using LSTM-based deep learning models. The model predicts direction classification (up/down) and closing price regression for the next 5 days (Monday-Friday).

## 🎯 Project Objective
Facilitate investors by providing weekly predictions of forex market direction and closing price in a data-driven manner.

## 📌 Target Currency

- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- NZD/USD
  
## 💼 Business Understanding

### Problem Statements
- How can the system make a prediction that stocks will rise / fall for the next trading week (Monday to Friday)?
- How can the system make price predictions for the next trading week (Monday to Friday)?

### Goals
- Develop a classification model to make predictions from historical trading data.
- Develop a regression model to make predictions from historical trading data.

### Solution Statements
- Build an LSTM-based classification model to make up/down predictions for the next trading week (Monday to Friday).
- Build an LSTM-based regression model to make price predictions for the next trading week (Monday to Friday).

## 💾 Data Understanding
The historical data is sourced from Yahoo Finance using the [yfinance](https://pypi.org/project/yfinance/) library and covers five major currency pairs: EUR/USD, USD/JPY, AUD/USD, GBP/USD, and NZD/USD. The dataset are clean, no missing values or duplicates. Each dataset contains 2716 rows and 5 columns:  
- **Close**: Closing price  
- **High**: Highest price of the day  
- **Low**: Lowest price of the day  
- **Open**: Opening price  
- **Volume**: Trading volume
  

## 🔧 Pipeline Details

### 1. **Data Retrieval:**
   - Retrieve historical data from Yahoo Finance (via `yfinance`) by currency pair.

### 2. **Data Preprocessing:**
  1. Ensure index is date
  2. Removing the volume column
  3. Calculates the daily price change
  4. Classification of price changes
  5. Volatility classification
  6. Calculating RSI technical features. As an indicator of momentum and overbought/oversold detection
  7. Calculating the technical features of Moving Averages or the average of closing prices in the last 5 and 10 days (MA_5 and MA_10, because the prediction is for the short-term)
  8. Calculating MACD technical features. As a trend & momentum indicator, and detection of trend direction and strength.
  9. Drop NA rows (because the initial MA and RSI will be empty)

### 3. **Data Preparation**
  1. Create UP/DOWN Targets. t+1 to t+5 (Monday-Friday)
  2. Normalize numerical features using `MinMaxScaler`.
  3. One-hot encoding of category class [‘Price_Range_Class’, ‘High_Low_Range_Class’].
  4. Creating LSTM Sequence. To prepare the time-series data to be used as input for the LSTM (Long Short-Term Memory) model.
  5. Prepare data for classification and regression models
  6. Create a regression target (closing price)

### 4. **Modeling & Result:**
 Because the model is used to predict the direction of movement (up/down) and the closing price. So it is made with 2 approaches, namely:
    - **Classification:** Predict up/down for the next 5 days (`sigmoid`, binary output).
    - **Regression:** Predict the closing price for the next 5 days (MSE loss).
    
  ##### a. Classification
  1. Split data. With test_size=0.2
  2. Build Model. With arcithectur :
     
      ```python
      model_cls = Sequential([
      Input(shape=(X.shape[1], X.shape[2])),
      LSTM(64, return_sequences=True),
      Dropout(0.3),
      LSTM(32),
      Dropout(0.3),
      Dense(5, activation='sigmoid')  # output 5 hari
       ])
      ```

  4. Train Model. The model is trained with :
      - epoch= 100
      - batch_size= 32
      - callback=[checkpoint_cls]: save the model every epoch if val_loss improves.
  
  5. Evaluation
 Classification model was evaluated using the f1-score metric, which considers the balance between precision and recall on each prediction day (t+1 to t+5). This evaluation is done to see the model's performance in predicting the binary price movement direction (up or down) for the next 5 days.

  #### b. Regression
  1. Split data. With test_size=0.2
  2. Build Model. With arcithectur :

      ```python
      model_reg = Sequential([
          LSTM(64, return_sequences=False, input_shape=(X_train_reg.shape[1], X_train_reg.shape[2])),
          Dropout(0.2),
          Dense(32, activation='relu'),
          Dense(5)  # Output untuk Close_t+1 hingga Close_t+5
      ])
      ```

  3. Train Model. The model is trained with :
      - validation_split=0.2,
      - epoch= 100
      - batch_size= 32
      - callback=[checkpoint_cls]: save the model every epoch if val_loss improves.
  
  4. Evaluation
 Regression models are tested using testing data with Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics.
  - **Mean Squared Error (MSE)**: calculates the average of the squared difference between the actual value (`y_i`) and the predicted value (`\hat{y}_i`). MSE penalizes larger errors due to the squaring of the difference. A smaller MSE value indicates a better model.

  ![MSE Evaluation Formula](https://i.ibb.co/VWh7cVJH/Screenshot-from-2025-06-09-13-09-41.png)

  - **Mean Absolute Error (MAE)**: calculates the average absolute value of the difference between the actual and predicted values. MAE gives an average idea of how far the model prediction is from the actual value in the same units as the original data.

  ![MAE Evaluation Formula](https://i.ibb.co/bkdbcdR/Screenshot-from-2025-06-09-13-11-23.png) 
  


## 📊 Evaluation Results
- **Classification**
```markdown
| Pair   | t+1 Accuracy | t+2 Accuracy | t+3 Accuracy | t+4 Accuracy | t+5 Accuracy |
| ------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| EURUSD | 59%          | 63%          | 65%          | 70%          | 73%          |
| GBPUSD | 64%          | 69%          | 74%          | 80%          | 81%          |
| USDJPY | 35%          | 35%          | 35%          | 35%          | 37%          |
| AUDUSD | 59%          | 64%          | 69%          | 72%          | 74%          |
| NZDUSD | 58%          | 68%          | 74%          | 77%          | 80%          |
```

- **Regression**
```markdown
| Pair   | Test Loss (MSE) | Test MAE  |
| ------ | --------------- | --------- |
| EURUSD | 0.000101        | 0.007523  |
| GBPUSD | 0.000132        | 0.009043  |
| USDJPY | 709.642212      | 26.132755 |
| AUDUSD | 0.000074        | 0.006665  |
| NZDUSD | 0.000113        | 0.008732  |
```

## 📄 Format Output JSON (example)
`EURUSD_prediksi.json`
```json
[
    {
        "t+1": {
            "classification": "Naik",
            "classification_probability": 0.6647449731826782,
            "predicted_close_price": 1.1417486126382725
        }
    },
    {
        "t+2": {
            "classification": "Turun",
            "classification_probability": 0.8859123587608337,
            "predicted_close_price": 1.1417473968990122
        }
    },
    {
        "t+3": {
            "classification": "Turun",
            "classification_probability": 0.9485937356948853,
            "predicted_close_price": 1.1416666718121233
        }
    },
    {
        "t+4": {
            "classification": "Turun",
            "classification_probability": 0.9733131527900696,
            "predicted_close_price": 1.141488791790627
        }
    },
    {
        "t+5": {
            "classification": "Naik",
            "classification_probability": 0.9799851775169373,
            "predicted_close_price": 1.141923887504211
        }
    }
]
```

## ✅ Conclusion
1. The models are able to predict both direction (classification) and closing prices (regression) effectively for most currency pairs.
2. USD/JPY shows weak performance in both classification and regression, indicating a need for further model refinement or feature engineering.
3. Other pairs such as GBP/USD, AUD/USD, and NZD/USD demonstrate solid and reliable prediction performance.
4. These results support the project's goal to provide data-driven forex insights for short-term trading decisions.



