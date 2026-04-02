# 🔊 Noise Analysis Tool

A comprehensive real-time noise monitoring, analysis, and prediction system built with **Streamlit, Pandas, Matplotlib, and Scikit-learn**.

## ✨ Features

### 1. **Real-time Dashboard**
- Live noise level monitoring
- Key metrics (Average, Peak, Min, Standard Deviation)
- Interactive location-based filtering

### 2. **Time Series Analysis** (Tab 1)
- Noise trends visualization with warning thresholds (70dB & 85dB)
- Hourly noise heatmap by day of week
- Multi-location comparison

### 3. **ML Predictions** (Tab 2)
- **Random Forest Regressor** for noise forecasting
- Predict noise levels for next 6-48 hours
- Confidence intervals for predictions
- Real-time model training

### 4. **Anomaly Detection** (Tab 3)
- **Isolation Forest** algorithm for outlier detection
- Identifies unusual noise events
- Anomaly scoring and location-based analysis

### 5. **Pattern Clustering** (Tab 4)
- **K-Means clustering** for hour-based patterns
- Classifies hours as Quiet, Moderate, or Noisy
- Smart recommendations for each pattern

## 🛠 Technologies Used

| Library | Purpose |
|---------|---------|
| Streamlit | Web application framework |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning (Random Forest, Isolation Forest, K-Means) |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher

### Step-by-Step Installation

1. **Clone or download the repository**
```bash
git clone https://github.com/yourusername/noise-analysis-tool.git
cd noise-analysis-tool