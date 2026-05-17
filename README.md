# Real Estate Price Prediction Model

A machine learning project that predicts house prices using Gradient Boosting and other regression algorithms.

## 📊 Project Overview

This project analyzes real estate data and builds a predictive model to estimate house prices based on location, age, accessibility, and other features.

**Best Model:** Gradient Boosting
- **R² Score:** 0.6206 (62% variance explained)
- **RMSE:** 7.26
- **MAE:** 4.78

## 🎯 Features

- **Data Preprocessing:** Feature engineering, outlier detection, and removal
- **Feature Engineering:**
  - Distance to city center (geographic calculation)
  - Accessibility score (convenience stores / MRT distance)
  - Age categories (new, young, middle, old)
  - Transaction season extraction
  - High store density indicator

- **Model Comparison:**
  - Linear Regression
  - Ridge Regression (L2)
  - Lasso Regression (L1)
  - Random Forest
  - **Gradient Boosting** ⭐ (Best performer)

- **Evaluation Metrics:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
  - Cross-validation scores

## 📁 Project Structure

```
linear-regression/
├── regression.ipynb           # Main analysis notebook
├── Real\ estate.csv          # Dataset
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── models/                   # (Production) Saved models
    ├── gradient_boosting_model.pkl
    ├── scaler.pkl
    └── preprocessing_config.pkl
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/linear-regression.git
cd linear-regression
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebook:
```bash
jupyter notebook regression.ipynb
```

## 📊 Dataset

- **Source:** Real estate transaction data
- **Rows:** 411 (after outlier removal)
- **Features:** 13 engineered features
- **Target:** House price per unit area

### Features

| Feature | Description |
|---------|-------------|
| X2 house age | Age of the house in years |
| X3 distance to MRT | Distance to nearest MRT station |
| X4 convenience stores | Number of nearby convenience stores |
| X5 latitude | Geographic latitude |
| X6 longitude | Geographic longitude |
| distance_to_city_center | Calculated Euclidean distance |
| accessibility_score | Composite accessibility metric |
| transaction_month | Month of transaction (1-12) |
| transaction_season | Season (0=Q1, 1=Q2, 2=Q3, 3=Q4) |
| high_store_density | Binary flag for 5+ stores |
| age_category_* | Categorical age bins (one-hot encoded) |

## 📈 Model Performance

### Gradient Boosting (Best Model)
```
MSE:  52.70
RMSE: 7.26
MAE:  4.78
R²:   0.6206
CV R²: 0.8057 (±0.0472)
```

### Prediction Statistics
- Mean Absolute Error: 4.78 price units
- Mean Absolute Percentage Error: 17.16%
- Max Overprediction: -30.31
- Max Underprediction: 29.99

## 🔍 Outlier Detection

- **Method:** Interquartile Range (IQR)
- **Outliers Removed:** 3 records
- **Original Dataset:** 414 rows
- **Final Dataset:** 411 rows

## 📚 Methodology

1. **Data Loading & Exploration** - Load CSV, check data types and missing values
2. **Feature Engineering** - Create new features for better predictions
3. **Outlier Detection** - Remove extreme values using IQR method
4. **Train-Test Split** - 80/20 split for validation
5. **Scaling** - StandardScaler normalization
6. **Model Training** - Train 5 different regression models
7. **Model Evaluation** - Compare metrics and select best
8. **Visualization** - Create diagnostic plots

## 🛠️ Technologies Used

- **Python 3.9+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **Jupyter** - Interactive notebooks

## 📋 Requirements

See `requirements.txt` for all dependencies:
```
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
```

## 🚀 Production Deployment

For production deployment, see [PRODUCTION.md](PRODUCTION.md):

- Model serialization and versioning
- REST API setup (FastAPI)
- Docker containerization
- Testing suite
- CI/CD pipeline
- Monitoring and logging

## 📝 Usage Example

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare input
new_data = pd.DataFrame({
    'X2 house age': [10],
    'X3 distance to the nearest MRT station': [500],
    # ... other features
})

# Make prediction
prediction = model.predict(scaler.transform(new_data))
print(f"Predicted price: {prediction[0]:.2f}")
```

## 📊 Visualization

The notebook includes:
- Actual vs Predicted scatter plot
- Residual analysis
- Error distribution histogram
- Model comparison charts

## 🐛 Troubleshooting

**Module not found error:**
```bash
pip install -r requirements.txt
```

**Jupyter not starting:**
```bash
python -m jupyter notebook
```

**Data not loading:**
- Ensure `Real estate.csv` is in the project root
- Check file path in the notebook

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering)


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Include error logs and steps to reproduce

## 🎯 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Additional feature engineering
- [ ] Time series analysis for temporal patterns
- [ ] REST API deployment
- [ ] Real-time prediction service
- [ ] Model retraining pipeline
- [ ] Data drift detection

---

**Last Updated:** May 17, 2026
#
