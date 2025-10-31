# 🏆 Advanced Price Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-yellow.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Competition-Top%2042%25-brightgreen.svg)](https://github.com/yourusername/price-prediction)

> 🎯 **Competition Result:** SMAPE 58.66 | **Rank:** Top 2500/6000 (Top 42%)

An ensemble machine learning pipeline for predicting product prices from catalog text descriptions. This solution combines advanced feature engineering, multiple text vectorization techniques, and stacked ensemble modeling to achieve competitive performance.

---

## 📊 Performance Metrics

| Metric | Value | Percentile |
|--------|-------|------------|
| **SMAPE** | 58.6571% | Top 42% |
| **Rank** | < 2500 | 6000 participants |
| **Models Used** | 3 + Meta | Stacked Ensemble |
| **Runtime** | ~12 min | CPU-optimized |

---

## ✨ Key Features

- 🔧 **Advanced Feature Engineering**: Extracts 20+ features from text (quantities, quality indicators, materials, sizes)
- 📝 **Multi-Strategy Text Vectorization**: TF-IDF with word/character n-grams + SVD dimensionality reduction
- 🤖 **Ensemble Learning**: XGBoost + LightGBM + Extra Trees with Ridge meta-model
- 🎯 **Competition-Safe**: No external LLMs or APIs - pure ML approach
- ⚡ **Efficient**: Optimized for CPU, runs in ~12 minutes
- 📚 **Well-Documented**: Extensive comments explaining every concept

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pandas
numpy
scikit-learn
xgboost
lightgbm
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ramoware/Amazon-ML-Challenge-2025.git
cd price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Ensure your data files are in the same directory
# Required files: train.csv, test.csv

# Run the pipeline
python model.py

# Output: output.csv
```

---

## 📁 Project Structure

```
price-prediction/
│
├── model.py             # Main pipeline script
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── train.csv(xlsx)      # Training data (not included)
├── test.csv(xlsx)       # Test data (not included)
└── output.csv(xlsx)     # Generated predictions
```

---

## 🧠 Methodology

### 1. Feature Engineering (20+ Features)

The pipeline extracts comprehensive features from product descriptions:

```python
# Quantity Detection
"Pack of 12" → quantity=12, is_multi_pack=1

# Quality Indicators  
"Premium Leather" → premium_count=1, net_quality=+1

# Size Scoring
"XXL Jumbo" → max_size_score=3

# Material Analysis
"Stainless Steel" → premium_material_count=2
```

**Feature Categories:**
- 📦 Quantity & Pack Size
- ⭐ Quality Level (Premium vs Economy)
- 📏 Size Indicators
- 🛠️ Material Composition
- 🔢 Numeric Patterns
- 📝 Text Complexity
- 🏷️ Brand Indicators

### 2. Text Vectorization

Multiple complementary approaches capture different aspects of text:

| Method | N-grams | Features | Purpose |
|--------|---------|----------|---------|
| **TF-IDF (Word 1-2)** | 1-2 | 200 → 50 | Semantic meaning |
| **TF-IDF (Word 1-3)** | 1-3 | 150 → 50 | Contextual phrases |
| **TF-IDF (Char 3-5)** | 3-5 | 100 → 50 | Spelling patterns |
| **Count (Char 3-6)** | 3-6 | 100 | Robust to typos |

**Total Text Features:** 250 dimensions

### 3. Ensemble Architecture

```
┌─────────────┐
│   XGBoost   │──┐
└─────────────┘  │
                 │     ┌──────────────┐
┌─────────────┐  ├───▶│  Ridge Meta   │──▶ Final Prediction
│  LightGBM   │──┤     │    Model     │
└─────────────┘  │     └──────────────┘
                 │
┌─────────────┐  │
│ Extra Trees │──┘
└─────────────┘
```

**Why This Works:**
- 🎯 **Diversity**: Different algorithms capture different patterns
- 🛡️ **Robustness**: Ensemble reduces overfitting
- 📈 **Performance**: Stacking often beats individual models

### 4. Post-Processing

Ensures realistic predictions:
- ✅ Minimum price: $0.01
- 📊 Maximum cap: 120% of training 98th percentile
- 🔧 Log-space smoothing (98% factor)

---

## 📈 Results Analysis

### Competition Performance

```
╔══════════════════════════════════════╗
║  SMAPE: 58.6571%                     ║
║  Rank: < 2500 / 6000                 ║
║  Percentile: Top 42%                 ║
╚══════════════════════════════════════╝
```

### Individual Model Performance

| Model | Validation SMAPE | Training Time |
|-------|------------------|---------------|
| XGBoost | ~60% | ~2 min |
| LightGBM | ~61% | ~1.5 min |
| Extra Trees | ~63% | ~1 min |
| **Stacked Ensemble** | **~59%** | **~5 min total** |

### Key Insights

1. **Feature Engineering Impact**: Engineered features provide 40% of predictive power
2. **Text Embeddings**: Character-level n-grams surprisingly effective
3. **Ensemble Benefit**: ~2% improvement over best single model
4. **Outlier Removal**: Removing top 2% prices improved stability

---

## 🔧 Hyperparameters

### XGBoost
```python
n_estimators=500      # More trees = better fit
learning_rate=0.05    # Conservative learning
max_depth=8           # Moderate complexity
subsample=0.8         # 80% data per tree
```

### LightGBM
```python
n_estimators=500
learning_rate=0.05
max_depth=8
subsample=0.8
```

### Extra Trees
```python
n_estimators=100      # Fewer trees (faster)
max_depth=20          # Deeper trees
min_samples_split=10  # Regularization
```

### Ridge Meta-Model
```python
alpha=1.0             # L2 regularization strength
```

---

## 🎯 Future Improvements

### Potential Enhancements (Not Implemented)

- [ ] **Advanced NLP**: Add sentiment analysis, POS tagging
- [ ] **Deep Learning**: BERT/transformer embeddings
- [ ] **Feature Selection**: Recursive feature elimination
- [ ] **Hyperparameter Tuning**: Bayesian optimization
- [ ] **Cross-Validation**: K-fold for robust validation
- [ ] **Neural Network**: Add deep learning to ensemble
- [ ] **Category-Specific Models**: Separate models per product category
- [ ] **Price Bucketing**: Classification + regression hybrid

**Expected Improvements:** 5-10% SMAPE reduction possible

---

## 📊 Visualization Ideas

### Feature Importance
```python
import matplotlib.pyplot as plt

# Plot XGBoost feature importance
xgb.plot_importance(xgb_model, max_num_features=20)
plt.title("Top 20 Most Important Features")
plt.show()
```

### Prediction Distribution
```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_clean, bins=50, edgecolor='black')
plt.title("Actual Price Distribution")

plt.subplot(1, 2, 2)
plt.hist(final_pred, bins=50, edgecolor='black')
plt.title("Predicted Price Distribution")

plt.tight_layout()
plt.show()
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: Memory Error**
```python
# Reduce feature dimensions
max_features=100  # Instead of 200
n_components=30   # Instead of 50
```

**Issue 2: Slow Training**
```python
# Reduce ensemble size
n_estimators=200  # Instead of 500
```

**Issue 3: Poor Performance**
```python
# Check data quality
print(train['price'].describe())
print(train['catalog_content'].isnull().sum())
```

---

## 📚 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
```

Install all at once:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

### Areas for Contribution
- Additional feature engineering ideas
- Alternative text vectorization methods
- Hyperparameter optimization experiments
- Documentation improvements
- Bug fixes

---

## 📖 Learning Resources

Want to understand the concepts better?

- **Ensemble Learning**: [Sklearn Ensemble Guide](https://scikit-learn.org/stable/modules/ensemble.html)
- **TF-IDF**: [Understanding TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/)
- **XGBoost**: [Official XGBoost Tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- **SMAPE**: [Forecast Error Metrics](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
- **Stacking**: [Stacked Generalization](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍👩‍👧‍👦 Team

**Ramdev Chaudhary [Team Leader]**
- GitHub: [@ramoware](https://github.com/ramoware)
- LinkedIn: [ramdevchaudhary](https://linkedin.com/in/ramdevchaudhary)
- Email: ramoware@gmail.com

**Pranita Jagtap [Co-Leader]**
- GitHub: [@PranitaJagtap](https://github.com/PranitaJagtap)
- LinkedIn: [PranitaJagtap](https://linkedin.com/in/PranitaJagtap)
- Email: jagtappranita2003@gmail.com

**Vedant Wadekar [Associate]**
- GitHub: [@Vedantwadekar2112](https://github.com/Vedantwadekar2112)
- LinkedIn: [vedant-wadekar-394948378](https://linkedin.com/in/vedant-wadekar-394948378)
- Email: vedantwadekar49@gmail.com

**Sony Yadav [Associate]**
- GitHub: [@ramoware](https://github.com/ramoware)
- LinkedIn: [sony-yadav-17393232a](https://linkedin.com/in/sony-yadav-17393232a)
- Email: soniy11265@gmail.com

---

## 🙏 Acknowledgments

- Competition organizers for the dataset and challenge
- scikit-learn, XGBoost, and LightGBM communities
- Open source ML community for inspiration

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/price-prediction&type=Date)](https://star-history.com/#yourusername/price-prediction&Date)

---

## 📞 Support

- 📧 Email: ramoware@gmail.com
- 💬 Issues: [GitHub Issues](https://github.com/ramoware/price-prediction/issues)
- 📖 Documentation: [Wiki](https://github.com/ramoware/price-prediction/wiki)

---

<div align="center">

**Made with ❤️ and ☕ for the ML community**

[⬆ Back to Top](#-advanced-price-prediction-model)

</div>
