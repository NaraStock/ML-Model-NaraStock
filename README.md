# NaraStock-ML

NaraStock is a forex currency price movement prediction project (EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD) using LSTM-based deep learning models. The model predicts direction classification (up/down) and closing price regression for the next 5 days (Monday-Friday).

---

## 🎯 Project Objective
Facilitate investors by providing weekly predictions of forex market direction and closing price in a data-driven manner.

---

## 💾 Dataset
The historical data is sourced from Yahoo Finance using the [yfinance](https://pypi.org/project/yfinance/) library and covers five major currency pairs: EUR/USD, USD/JPY, AUD/USD, GBP/USD, and NZD/USD. Each dataset contains 2716 rows and 5 columns:  
- **Close**: Closing price  
- **High**: Highest price of the day  
- **Low**: Lowest price of the day  
- **Open**: Opening price  
- **Volume**: Trading volume  
---

## 📁 Folder Structure

```
NaraStock-ML/
│
├── README.md                  
├── requirements.txt           
│
├── notebooks/                 
│   ├── EURUSD_notebook.ipynb
│   ├── GBPUSD_notebook.ipynb
│   ├── USDJPY_notebook.ipynb
│   ├── AUDNZD_notebook.ipynb
│   └── NZDUSD_notebook.ipynb
│
├── outputs/                   
│   ├── EURUSD_prediksi.json
│   ├── GBPUSD_prediksi.json
│   ├── USDJPY_prediksi.json
│   ├── AUDUSD_prediksi.json
│   └── NZDUSD_prediksi.json
│
└── docs/                      
    └── documentation.txt
```

---

## ⚙️ How to use

1. Clone this repo
2. Open and run the notebooks in each currency pair folder, or use the inference script provided.
3. Make sure to install the required libraries:
4. Notebook containing load library, load data, EDA, preprocessing, preparation, training, evaluation, and inference.
5. The prediction output will be stored in a JSON file.

---

## 🔧 Pipeline Details

### Preprocessing & Preparation
- Data retrieval from Yahoo Finance
- Technical features: RSI, MA, MACD, etc.
- Data normalization with MinMaxScaler
- Data sequence preparation for LSTM

### Modeling
- Classification model (sigmoid, 5-day output)
- Regression model (MSE loss, 5-day output)

### Evaluation
- Classification: F1-score and accuracy per day
- Regression: MSE and MAE

---

## 📊 Evaluation Results
- **Classification**
```markdown
| Pair   | t+1 Accuracy | t+2 Accuracy | t+3 Accuracy | t+4 Accuracy | t+5 Accuracy |
|--------|--------------|--------------|--------------|--------------|--------------|
| EURUSD | 59%          | 63%          | 65%          | 70%          | 73%          |
| GBPUSD | 64%          | 69%          | 74%          | 80%          | 81%          |
| USDJPY | 35%          | 35%          | 35%          | 35%          | 37%          |
| AUDUSD | 59%          | 64%          | 69%          | 72%          | 74%          |
| NZDUSD | 58%          | 68%          | 74%          | 77%          | 80%          |
```

- **Regression**
```markdown
| Pair   | Test Loss (MSE) | Test MAE  |
|--------|------------------|-----------|
| EURUSD | 0.000101         | 0.007523  |
| GBPUSD | 0.000132         | 0.009043  |
| USDJPY | 709.642212       | 26.132755 |
| AUDUSD | 0.000074         | 0.006665  |
| NZDUSD | 0.000113         | 0.008732  |
```


---

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
