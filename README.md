# Quora Question Pairs Duplicate Detection

A machine learning project to detect duplicate question pairs from Quora using advanced NLP techniques and feature engineering.

## 📋 Project Overview

This project identifies whether two questions are duplicates using a Gradient Boosting classifier trained on **1,00,000 question pairs** from the Quora dataset. The model combines preprocessing, traditional NLP features, and fuzzy matching to achieve high accuracy.

## 🎯 Problem Statement

Quora has millions of questions. Many questions are duplicates that need to be consolidated. This project builds a model to automatically detect duplicate question pairs.

## 📊 Dataset

- **Source:** Quora Question Pairs Competition Dataset
- **Total Samples:** 100000 question pairs
- **Features:** question1, question2, is_duplicate (label)
- **Class Distribution:** ~37% duplicates, ~63% non-duplicates

## 🏗️ Project Architecture

```
quora-question-pairs/
├── initial_EDA.ipynb                              # Exploratory Data Analysis
├── bow-with-basic-features.ipynb                  # Basic feature engineering
├── bow-with-preprocessing-and-advanced-features.ipynb  # Advanced features
├── memory-efficient-training.ipynb          # Trained with 22 handcrafted features only
|__tfidf-enhanced-training.ipynb             # Trained with 22 + TF-IDF(500) features
├── app.py                                        # Streamlit deployment app
├── helper.py                                     # Feature engineering functions
├── create_stopwords.py                           # Stopwords setup
├── model_hybrid_tfidf.pkl                                    # final_trained model
├── stopwords.pkl                                 # Cached stopwords
├── tfidf_vectorizer.pkl                          # countvectorizer for TF-IDF
├── train.csv                                     # Dataset
└── requirements.txt                              # Dependencies
```

## 🔬 Feature Engineering

### **22+500=522 Total Features**

#### 1. **Basic Features** (7)
- Question 1 character length
- Question 2 character length
- Question 1 token count
- Question 2 token count
- Common words between questions
- Total unique words
- Common words ratio

#### 2. **Token-Based Features** (8)
- Common non-stopword ratio (min/max across questions)
- Common stopword ratio (min/max)
- Common token ratio (min/max)
- Same first word indicator
- Same last word indicator

#### 3. **Length-Based Features** (3)
- Absolute length difference
- Average token length
- Longest common substring ratio

#### 4. **Fuzzy Matching Features** (4)
- Fuzz Ratio (character-level similarity)
- Fuzz Partial Ratio
- Token Sort Ratio (order-independent)
- Token Set Ratio (ignores duplicates)

#### 5. **TF-IDF FEATURES**
- Total 500 tfidf features(250 for each question)
## 🛠️ Preprocessing Pipeline

1. **Lowercasing** - Normalize case
2. **Special Character Handling** - Replace $, %, ₹, €, @, [math]
3. **Number Normalization** - Convert 1000 → 1k, 1000000 → 1m
4. **Contraction Expansion** - "can't" → "can not", "he's" → "he is"
5. **HTML Cleanup** - Remove tags using BeautifulSoup
6. **Punctuation Removal** - Keep only alphanumeric and spaces
7. **Stopword Identification** - Using NLTK English stopwords

## 📈 Model Performance

### **Test Set Results** (25% of data)

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7842 |
| **Precision** | 0.7128 |
| **Recall** | 0.7011 |
| **F1 Score** | 0.7069 |
| **ROC-AUC** | 0.8713 |

### **Cross-Validation** (5-Fold Stratified)
- Mean F1: 0.70 ± 0.02

### **Model Status**
✅ Low overfitting | ✅ Generalizes well | ✅ Production ready

## 🚀 Usage

### 1. **Installation**

```bash
# Create virtual environment
python -m venv question_venv
.\question_venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run Streamlit App**

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 3. **Programmatic Usage**

```python
import pickle
import helper

# Load model
model = pickle.load(open('model_hybrid_tfidf.pkl', 'rb'))

# Create query
q1 = "What is Python?"
q2 = "Python programming language explained"
query = helper.query_point_creator(q1, q2)

# Predict
prediction = model.predict(query)[0]
print("Duplicate!" if prediction == 1 else "Not Duplicate!")
```

## 📚 Notebooks

### **1. initial_EDA.ipynb**
- Dataset exploration
- Distribution analysis
- Text statistics
- Class imbalance assessment

### **2. bow-with-basic-features.ipynb**
- Bag of Words vectorization
- Basic statistical features

### **3. bow-with-preprocessing-and-advanced-features.ipynb**
- Full preprocessing pipeline
- Advanced feature engineering
- Feature correlation analysis
- Model training with 30K samples

### **4. model_hybrid_tfidf.ipynb** ⭐ **NEW**
- **Uses  100k samples**
- **3-Fold Stratified Cross-Validation**
- **Gradient Boosting**
- **Complete evaluation metrics**
- **Production-ready model**

## 🔧 Model Details

### **Algorithm:** Gradient Boosting Classifier

```python
GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,    # Shrinkage parameter
    max_depth=6,           # Tree depth limit
    min_samples_split=20,  # Minimum samples to split
    subsample=0.8,         # Stochastic gradient boosting
    random_state=42        # Reproducibility
    verbose=1
)
```

### **Why Gradient Boosting?**
- Handles mixed feature types well
- Captures non-linear relationships
- Built-in feature importance
- Better than Random Forest for this task

## 📊 Key Findings

1. **Fuzzy matching features are most important** - Capture string similarity well
2. **Token-based features provide discriminative power** - Differentiate duplicate patterns
3. **Length-based features less important individually** - But useful in ensemble

## ⚠️ Limitations & Future Work

### **Current Limitations**
- English-only (no multilingual support)
- No semantic embeddings (BERT, Word2Vec)
- Fixed feature set (no dynamic features)

### **Future Improvements**
- [ ] Add BERT embeddings for semantic similarity
- [ ] Implement ensemble methods (Stack multiple models)
- [ ] Hyperparameter grid search
- [ ] Class weight adjustment for imbalanced data
- [ ] SMOTE for synthetic minority oversampling
- [ ] API deployment (FastAPI/Flask)
- [ ] Multilingual support

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `helper.py` | Core feature engineering functions |
| `app.py` | Streamlit web application |
| `create_stopwords.py` | Setup script for stopwords cache |
| `model_hybrid_tfidf.pkl` | Trained classifier (production) |
| `tfidf_vectorizer.pkl` | Cached tfidfVectorizer |
| `stopwords.pkl` | Cached English stopwords |
| `requirements.txt` | Python dependencies |

## 💾 Dependencies

```
scikit-learn >= 1.8.0
pandas >= 2.3.3
numpy >= 2.4.2
scipy >= 1.17.1
nltk >= 3.9.3
fuzzywuzzy >= 0.18.0
python-Levenshtein >= 0.27.3
streamlit >= 1.54.0
beautifulsoup4 >= 4.14.3
```

## 📈 Performance Comparison

| Approach | F1 Score | Notes |
|----------|----------|-------|
| BoW + Basic Features | 0.76 |
| BoW + Advanced Features | 0.75 | Fuzzy matching added |
| **Tfidf + 100k data** | **0.78** | ✅ **Current Production Model** |

## 🎓 Learning Outcomes

This project demonstrates:

1. **End-to-End ML Pipeline**
   - Data exploration → Feature engineering → Model training → Evaluation → Deployment

2. **Feature Engineering for NLP**
   - Preprocessing techniques
   - Statistical features
   - String similarity metrics
   - Fuzzy matching

3. **Rigorous Model Evaluation**
   - Train-test stratified split
   - Cross-validation
   - Multiple evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
   - Confusion matrix analysis

4. **Production Deployment**
   - Pickle serialization
   - Web interface (Streamlit)
   - Real-time predictions

5. **Best Practices**
   - Feature scaling
   - Stratified sampling

## 🚀 Deployment

### **Local Streamlit**
```bash
streamlit run app.py
```

### **Production API (Future)**
```bash
uvicorn api:app --reload  # FastAPI
```

## 📞 Contact & Attribution

- **Dataset:** Kaggle Quora Question Pairs
- **Author:** Durgesh Kumar
- **Date:** 06 Mar 2026

## ✨ Quick Start

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# 2. Train Model (if needed)
jupyter notebook tfidf-enhanced-training.ipynb

# 3. Run App
streamlit run app.py

# 4. Test
# Input two questions and see if they're duplicates!
```

---

**Last Updated:** March 6, 2026  
**Model Version:** 3.0 (100k Dataset, CV-Validated)  
**Status:** ✅ Production Ready
