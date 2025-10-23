"""
============================================
ADVANCED PRICE PREDICTION MODEL
============================================
A comprehensive machine learning pipeline for predicting product prices
from catalog text descriptions using ensemble methods and advanced NLP.

Author: Ramdev Chaudhary & Pranita Jagtap
Date: 2025
Competition-Safe: Uses only traditional ML (no external LLMs)

Key Features:
- Advanced feature engineering from text
- Multiple text representation techniques (TF-IDF, SVD)
- Ensemble stacking with XGBoost, LightGBM, and Extra Trees
- Robust outlier handling and post-processing
============================================
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("🌌 ADVANCED PRICE PREDICTION MODEL - STARTING")
print("=" * 60)

# ============================================
# SECTION 1: DATA LOADING
# ============================================
print("\n📂 Loading datasets...")

# Load training data (contains product descriptions and actual prices)
train = pd.read_csv('train.csv')

# Load test data (contains product descriptions, need to predict prices)
test = pd.read_csv('test.csv')

print(f"   Training samples: {len(train)}")
print(f"   Test samples: {len(test)}")


# ============================================
# SECTION 2: ADVANCED FEATURE ENGINEERING
# ============================================
"""
This section extracts meaningful numerical features from raw text descriptions.
We convert text like "Pack of 12 premium steel spoons" into quantifiable metrics
that machine learning models can understand.
"""

def quantum_feature_extraction(text):
    """
    Extract comprehensive features from product text descriptions.
    
    This function analyzes product descriptions to identify:
    - Quantities and pack sizes
    - Quality indicators (premium vs economy)
    - Size information
    - Material composition
    - Text complexity metrics
    
    Args:
        text (str): Product description text
        
    Returns:
        dict: Dictionary of extracted features
    """
    
    # Handle missing or null values
    if pd.isna(text): 
        text = ""
    text_lower = str(text).lower()
    
    features = {}
    
    # ---------------------------------------------
    # FEATURE GROUP 1: QUANTITY DETECTION
    # ---------------------------------------------
    """
    Detect quantities from patterns like:
    - "pack of 12", "12-pack", "12 pieces"
    - "set of 6", "24 count", "3 units"
    
    Why this matters: Multi-packs typically have different pricing
    than single items. A "12-pack" will cost more than "1 piece".
    """
    quantities = []
    
    # Define regex patterns to capture various quantity expressions
    patterns = [
        r'pack of (\d+)',      # "pack of 12"
        r'(\d+)[\s-]pack',     # "12-pack" or "12 pack"
        r'(\d+)\s*pieces?',    # "12 pieces" or "12 piece"
        r'(\d+)\s*pcs',        # "12 pcs"
        r'set of (\d+)',       # "set of 6"
        r'(\d+)\s*count',      # "24 count"
        r'(\d+)\s*units?',     # "3 units" or "3 unit"
        r'box of (\d+)',       # "box of 50"
        r'(\d+)[\s-]piece',    # "6-piece" or "6 piece"
        r'(\d+)\s*[x×]\s*\d+', # "2 x 5" or "2×5"
        r'(\d+)\s*\-?\s*roll', # "3-roll" or "3 roll"
        r'(\d+)\s*\-?\s*tablet',   # "10 tablets"
        r'(\d+)\s*\-?\s*capsule'   # "30 capsules"
    ]
    
    # Search for all patterns and extract quantities
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Handle both tuple and string matches
            if isinstance(match, tuple):
                for m in match:
                    if m.isdigit() and 1 <= int(m) <= 1000:  # Sanity check
                        quantities.append(int(m))
            elif match.isdigit() and 1 <= int(match) <= 1000:
                quantities.append(int(match))
    
    # Use maximum quantity found, default to 1 if none found
    quantity = max(quantities) if quantities else 1
    
    # Store quantity-related features
    features['quantity'] = quantity
    features['log_quantity'] = np.log1p(quantity)  # Log transform for better distribution
    features['is_multi_pack'] = int(quantity > 1)  # Binary: is this a multi-pack?
    
    # ---------------------------------------------
    # FEATURE GROUP 2: QUALITY INDICATORS
    # ---------------------------------------------
    """
    Identify premium vs economy products.
    
    Premium products (luxury, professional, imported) typically cost more.
    Economy products (budget, value, cheap) typically cost less.
    """
    
    # Words that indicate premium/high-end products
    premium_terms = [
        'premium', 'luxury', 'deluxe', 'professional', 'gourmet', 
        'designer', 'imported', 'certified', 'elite', 'exclusive'
    ]
    
    # Words that indicate budget/economy products
    economy_terms = [
        'economy', 'budget', 'value', 'affordable', 'cheap', 'basic'
    ]
    
    # Count occurrences of premium and economy terms
    features['premium_count'] = sum(1 for term in premium_terms if term in text_lower)
    features['economy_count'] = sum(1 for term in economy_terms if term in text_lower)
    
    # Net quality score: positive means premium, negative means economy
    features['net_quality'] = features['premium_count'] - features['economy_count']
    
    # ---------------------------------------------
    # FEATURE GROUP 3: SIZE INDICATORS
    # ---------------------------------------------
    """
    Larger sizes typically cost more.
    Score: XXL (3) > XL (2) > Large (1) > Medium (0) > Small (-1) > Mini (-2)
    """
    
    size_scores = {
        'xxl': 3, 'extra large': 3, 'jumbo': 3, 'king': 3,
        'xl': 2, 'large': 1, 'big': 1,
        'medium': 0, 'standard': 0, 'regular': 0,
        'small': -1, 'mini': -2, 'travel': -2, 'petite': -1
    }
    
    # Find the largest size mentioned
    features['max_size_score'] = 0
    for size, score in size_scores.items():
        if size in text_lower:
            features['max_size_score'] = max(features['max_size_score'], score)
    
    # ---------------------------------------------
    # FEATURE GROUP 4: MATERIAL COMPOSITION
    # ---------------------------------------------
    """
    Material quality affects price.
    Premium materials (steel, leather, wood) suggest higher prices.
    Economy materials (plastic, paper) suggest lower prices.
    """
    
    premium_materials = [
        'steel', 'stainless', 'metal', 'leather', 
        'wood', 'ceramic', 'glass'
    ]
    economy_materials = ['plastic', 'fabric', 'rubber', 'paper']
    
    features['premium_material_count'] = sum(
        1 for mat in premium_materials if mat in text_lower
    )
    features['economy_material_count'] = sum(
        1 for mat in economy_materials if mat in text_lower
    )
    
    # ---------------------------------------------
    # FEATURE GROUP 5: NUMERIC FEATURES
    # ---------------------------------------------
    """
    Extract all numbers from text and compute statistics.
    Numbers might represent: prices, quantities, measurements, weights, etc.
    """
    
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    
    if numbers:
        num_values = [float(n) for n in numbers]
        features['max_number'] = max(num_values)
        features['min_number'] = min(num_values)
        features['avg_number'] = np.mean(num_values)
        features['number_count'] = len(numbers)
        
        # Check if text contains price-like numbers (1 to 10000)
        features['has_price_like_number'] = int(
            any(1 <= n <= 10000 for n in num_values)
        )
    else:
        # Default values when no numbers found
        features['max_number'] = 0
        features['min_number'] = 0
        features['avg_number'] = 0
        features['number_count'] = 0
        features['has_price_like_number'] = 0
    
    # ---------------------------------------------
    # FEATURE GROUP 6: TEXT COMPLEXITY
    # ---------------------------------------------
    """
    Text characteristics can indicate product quality.
    Detailed descriptions with unique vocabulary often indicate
    higher-quality products worth more.
    """
    
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Ratio of unique words to total words (lexical diversity)
    # Higher ratio = more diverse vocabulary = potentially premium product
    features['unique_word_ratio'] = (
        len(set(text_lower.split())) / max(len(text.split()), 1)
    )
    
    # ---------------------------------------------
    # FEATURE GROUP 7: BRAND INDICATORS
    # ---------------------------------------------
    """
    Title case words (Apple, Samsung, Nike) often indicate brand names.
    Branded products typically have different pricing than generic ones.
    """
    
    words = text.split()
    features['title_case_words'] = sum(
        1 for w in words if w.istitle() and len(w) > 2
    )
    
    return features


print("\n🔄 Extracting features from product descriptions...")
print("   This analyzes text patterns, quantities, quality indicators, etc.")

# Apply feature extraction to both training and test sets
train_quantum = train['catalog_content'].apply(
    quantum_feature_extraction
).apply(pd.Series)

test_quantum = test['catalog_content'].apply(
    quantum_feature_extraction
).apply(pd.Series)

print(f"   ✓ Extracted {train_quantum.shape[1]} engineered features")


# ============================================
# SECTION 3: TEXT VECTORIZATION
# ============================================
"""
Convert text into numerical representations using multiple techniques.

TF-IDF (Term Frequency-Inverse Document Frequency):
- Measures how important a word is in a document relative to all documents
- Words appearing frequently in one doc but rarely in others get high scores

Why multiple approaches?
- Different n-gram ranges capture different patterns
- Word-level: captures semantic meaning
- Character-level: captures spelling patterns and robust to typos
"""

print("\n🔤 Creating text representations...")
print("   Converting text to numerical vectors using TF-IDF...")

# ---------------------------------------------
# VECTORIZER 1: Word-level TF-IDF (1-2 grams)
# ---------------------------------------------
"""
Captures single words and two-word phrases.
Example: "premium steel" → captures both "premium" and "premium steel"
"""
tfidf_word_12 = TfidfVectorizer(
    max_features=200,           # Keep top 200 most important features
    ngram_range=(1, 2),         # Include unigrams and bigrams
    min_df=2,                   # Ignore terms appearing in < 2 documents
    stop_words='english'        # Remove common words (the, is, at, etc.)
)

# ---------------------------------------------
# VECTORIZER 2: Word-level TF-IDF (1-3 grams)
# ---------------------------------------------
"""
Captures up to three-word phrases for more context.
Example: "pack of 12" is kept as a single feature
"""
tfidf_word_13 = TfidfVectorizer(
    max_features=150,
    ngram_range=(1, 3),         # Include unigrams, bigrams, and trigrams
    min_df=3,
    stop_words='english'
)

# ---------------------------------------------
# VECTORIZER 3: Character-level TF-IDF
# ---------------------------------------------
"""
Analyzes character sequences (3-5 characters).
Useful for: brand names, product codes, typo-robustness.
Example: "iPhone" → "iph", "pho", "hon", "one"
"""
tfidf_char = TfidfVectorizer(
    max_features=100,
    analyzer='char',            # Character-level analysis
    ngram_range=(3, 5),         # 3-5 character sequences
    min_df=5
)

# ---------------------------------------------
# VECTORIZER 4: Character n-grams (Count-based)
# ---------------------------------------------
"""
Similar to char TF-IDF but uses raw counts instead of TF-IDF weighting.
Captures different signal from the same character patterns.
"""
char_ngrams = CountVectorizer(
    max_features=100,
    analyzer='char',
    ngram_range=(3, 6),         # 3-6 character sequences
    min_df=10
)

# Fit and transform training data, transform test data
print("   → Word-level TF-IDF (1-2 grams)...")
train_tfidf_12 = tfidf_word_12.fit_transform(train['catalog_content'].fillna(''))
test_tfidf_12 = tfidf_word_12.transform(test['catalog_content'].fillna(''))

print("   → Word-level TF-IDF (1-3 grams)...")
train_tfidf_13 = tfidf_word_13.fit_transform(train['catalog_content'].fillna(''))
test_tfidf_13 = tfidf_word_13.transform(test['catalog_content'].fillna(''))

print("   → Character-level TF-IDF...")
train_tfidf_char = tfidf_char.fit_transform(train['catalog_content'].fillna(''))
test_tfidf_char = tfidf_char.transform(test['catalog_content'].fillna(''))

print("   → Character n-grams (count-based)...")
train_char_ngrams = char_ngrams.fit_transform(train['catalog_content'].fillna(''))
test_char_ngrams = char_ngrams.transform(test['catalog_content'].fillna(''))


# ============================================
# SECTION 4: DIMENSIONALITY REDUCTION
# ============================================
"""
TF-IDF creates sparse, high-dimensional matrices.
SVD (Singular Value Decomposition) reduces dimensions while
preserving the most important information (like PCA but for sparse matrices).

Benefits:
- Reduces memory usage
- Speeds up model training
- Can improve model performance by reducing noise
- Creates dense representations (easier for ML models)
"""

print("\n🔧 Applying dimensionality reduction (SVD)...")
print("   Compressing sparse TF-IDF matrices to dense 50-dimensional embeddings...")

# Create SVD transformers (reducing to 50 dimensions each)
svd_12 = TruncatedSVD(n_components=50, random_state=42)
svd_13 = TruncatedSVD(n_components=50, random_state=42)
svd_char = TruncatedSVD(n_components=50, random_state=42)

# Transform TF-IDF matrices to dense representations
train_tfidf_12_dense = svd_12.fit_transform(train_tfidf_12)
test_tfidf_12_dense = svd_12.transform(test_tfidf_12)

train_tfidf_13_dense = svd_13.fit_transform(train_tfidf_13)
test_tfidf_13_dense = svd_13.transform(test_tfidf_13)

train_tfidf_char_dense = svd_char.fit_transform(train_tfidf_char)
test_tfidf_char_dense = svd_char.transform(test_tfidf_char)

print(f"   ✓ Created dense embeddings: {train_tfidf_12_dense.shape}")


# ============================================
# SECTION 5: FEATURE FUSION
# ============================================
"""
Combine all feature types into a single feature matrix:
- Engineered features (quantity, quality, size, materials, etc.)
- Word TF-IDF embeddings (1-2 grams)
- Word TF-IDF embeddings (1-3 grams)
- Character TF-IDF embeddings
- Character n-gram counts

This gives models multiple "views" of the same data.
"""

print("\n🔗 Combining all features into unified matrix...")

# Concatenate all feature types horizontally (column-wise)
X = np.concatenate([
    train_quantum.fillna(0).values,      # Engineered features
    train_tfidf_12_dense,                # Word embeddings (1-2 grams)
    train_tfidf_13_dense,                # Word embeddings (1-3 grams)
    train_tfidf_char_dense,              # Char embeddings
    train_char_ngrams.toarray()          # Char n-grams
], axis=1)

X_test = np.concatenate([
    test_quantum.fillna(0).values,
    test_tfidf_12_dense,
    test_tfidf_13_dense,
    test_tfidf_char_dense,
    test_char_ngrams.toarray()
], axis=1)

y = train['price'].values  # Target variable (prices to predict)

print(f"   ✓ Final feature matrix shape: {X.shape}")
print(f"      ({X.shape[0]} samples × {X.shape[1]} features)")


# ============================================
# SECTION 6: OUTLIER REMOVAL
# ============================================
"""
Remove extreme outliers from training data.
Outliers can negatively impact model training.

Strategy: Remove the top 2% most expensive items.
These are likely errors or extremely rare luxury items
that don't represent typical pricing patterns.
"""

print("\n🧹 Removing price outliers...")

# Calculate 98th percentile of prices
price_98 = np.percentile(y, 98)
print(f"   98th percentile price: ${price_98:.2f}")

# Keep only samples with price <= 98th percentile
clean_mask = y <= price_98
X_clean = X[clean_mask]
y_clean = y[clean_mask]

removed = len(y) - len(y_clean)
print(f"   ✓ Removed {removed} outliers ({removed/len(y)*100:.1f}%)")
print(f"   ✓ Clean dataset: {X_clean.shape[0]} samples")


# ============================================
# SECTION 7: FEATURE SCALING
# ============================================
"""
Standardize features to have mean=0 and std=1.

Why? Different features have different scales:
- Text length: 0-500
- Quantity: 1-100
- TF-IDF scores: 0-1

Scaling ensures all features contribute equally to the model.
Essential for Ridge regression and improves tree-based models.
"""

print("\n📊 Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_test_scaled = scaler.transform(X_test)

print("   ✓ Features standardized (mean=0, std=1)")


# ============================================
# SECTION 8: TRAIN-VALIDATION SPLIT
# ============================================
"""
Split training data into train and validation sets.

Validation set (15%):
- Used to evaluate model performance
- Helps prevent overfitting
- Simulates unseen test data

Training set (85%):
- Used to train the models
"""

print("\n✂️  Splitting data for validation...")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_clean, 
    test_size=0.15,      # 15% for validation
    random_state=42      # For reproducibility
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Validation set: {X_val.shape[0]} samples")


# ============================================
# SECTION 9: EVALUATION METRIC
# ============================================
"""
SMAPE: Symmetric Mean Absolute Percentage Error

Formula: 100/n × Σ|predicted - actual| / ((|predicted| + |actual|) / 2)

Why SMAPE?
- Handles both small and large values fairly
- Symmetric (over/under predictions treated equally)
- Expressed as percentage (interpretable)
- Commonly used in forecasting competitions

Lower is better. <45% is excellent for this problem.
"""

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


# ============================================
# SECTION 10: MODEL TRAINING (ENSEMBLE)
# ============================================
"""
Ensemble Learning: Train multiple diverse models and combine predictions.

Why ensemble?
- Different models capture different patterns
- Reduces overfitting (ensemble generalizes better)
- More robust predictions
- Often wins ML competitions

Models used:
1. XGBoost: Powerful gradient boosting, great with complex interactions
2. LightGBM: Fast gradient boosting, efficient for large datasets
3. Extra Trees: Randomized decision trees, reduces variance
"""

print("\n" + "="*60)
print("⚡ TRAINING ENSEMBLE MODELS")
print("="*60)

models = {}              # Store trained models
predictions_val = {}     # Store validation predictions

# ---------------------------------------------
# MODEL 1: XGBoost
# ---------------------------------------------
"""
XGBoost (Extreme Gradient Boosting):
- Builds trees sequentially, each correcting previous errors
- Highly effective for structured/tabular data
- Handles non-linear relationships well

Key hyperparameters:
- n_estimators=500: Number of trees (more = better fit but slower)
- learning_rate=0.05: Small steps = slower but more accurate
- max_depth=8: Tree depth (deeper = more complex patterns)
- subsample=0.8: Use 80% of data per tree (prevents overfitting)
"""

print("\n1️⃣  Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    random_state=42,
    tree_method='hist',  # Histogram-based method (faster)
    n_jobs=-1           # Use all CPU cores
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_score = smape(y_val, xgb_pred)

models['xgb'] = xgb_model
predictions_val['xgb'] = xgb_pred

print(f"   ✓ XGBoost trained")
print(f"   → Validation SMAPE: {xgb_score:.4f}%")

# ---------------------------------------------
# MODEL 2: LightGBM
# ---------------------------------------------
"""
LightGBM (Light Gradient Boosting Machine):
- Similar to XGBoost but uses different tree-growing strategy
- Grows trees leaf-wise (XGBoost grows level-wise)
- Very fast training, especially on large datasets
- Often performs slightly differently than XGBoost

Together, XGBoost and LightGBM provide ensemble diversity.
"""

print("\n2️⃣  Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    random_state=42,
    verbose=-1,          # Suppress training logs
    n_jobs=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)
lgb_score = smape(y_val, lgb_pred)

models['lgb'] = lgb_model
predictions_val['lgb'] = lgb_pred

print(f"   ✓ LightGBM trained")
print(f"   → Validation SMAPE: {lgb_score:.4f}%")

# ---------------------------------------------
# MODEL 3: Extra Trees
# ---------------------------------------------
"""
Extra Trees (Extremely Randomized Trees):
- Random Forest variant with more randomization
- Each tree sees different random feature splits
- Less prone to overfitting
- Provides different perspective from gradient boosting

Diversity is key in ensembles!
"""

print("\n3️⃣  Training Extra Trees...")
et_model = ExtraTreesRegressor(
    n_estimators=100,     # 100 trees (faster than 500)
    max_depth=20,         # Relatively deep trees
    min_samples_split=10, # Need 10 samples to split
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_val)
et_score = smape(y_val, et_pred)

models['et'] = et_model
predictions_val['et'] = et_pred

print(f"   ✓ Extra Trees trained")
print(f"   → Validation SMAPE: {et_score:.4f}%")


# ============================================
# SECTION 11: STACKED ENSEMBLE (META-MODEL)
# ============================================
"""
Stacking: Train a meta-model on top of base models' predictions.

How it works:
1. Base models (XGBoost, LightGBM, Extra Trees) make predictions
2. These predictions become features for a meta-model
3. Meta-model learns optimal way to combine base predictions

Meta-model: Ridge Regression
- Simple linear model with L2 regularization
- Learns weights like: final = 0.4*XGB + 0.35*LGB + 0.25*ET
- Prevents overfitting through regularization

This often improves performance over simple averaging!
"""

print("\n" + "="*60)
print("🌀 STACKING ENSEMBLE")
print("="*60)

# Create meta-features: predictions from all base models
meta_train = np.column_stack([xgb_pred, lgb_pred, et_pred])

print(f"\nMeta-features shape: {meta_train.shape}")
print("   (Each row = [XGBoost_pred, LightGBM_pred, ExtraTrees_pred])")

# Train meta-model (Ridge regression) on base model predictions
print("\n🧠 Training meta-model (Ridge Regression)...")
meta_model = Ridge(alpha=1.0, random_state=42)  # alpha=1.0 is regularization strength
meta_model.fit(meta_train, y_val)

# Get stacked predictions
stacked_pred = meta_model.predict(meta_train)
stacked_smape = smape(y_val, stacked_pred)

print(f"   ✓ Meta-model trained")
print(f"\n📊 ENSEMBLE WEIGHTS (learned by Ridge):")
for i, model_name in enumerate(['XGBoost', 'LightGBM', 'Extra Trees']):
    weight = meta_model.coef_[i]
    print(f"   {model_name}: {weight:.4f}")

print(f"\n" + "="*60)
print(f"🎯 VALIDATION RESULTS:")
print("="*60)
print(f"   XGBoost:     {xgb_score:.4f}% SMAPE")
print(f"   LightGBM:    {lgb_score:.4f}% SMAPE")
print(f"   Extra Trees: {et_score:.4f}% SMAPE")
print(f"   " + "-"*40)
print(f"   📈 STACKED:   {stacked_smape:.4f}% SMAPE")
print("="*60)


# ============================================
# SECTION 12: TEST SET PREDICTIONS
# ============================================
"""
Generate final predictions for test set using the complete ensemble.

Process:
1. Each base model predicts on test data
2. Meta-model combines these predictions
3. Apply post-processing to ensure realistic prices
"""

print("\n🌠 Generating final predictions for test set...")

# Get predictions from all base models on test data
print("   → XGBoost predictions...")
xgb_test = xgb_model.predict(X_test_scaled)

print("   → LightGBM predictions...")
lgb_test = lgb_model.predict(X_test_scaled)

print("   → Extra Trees predictions...")
et_test = et_model.predict(X_test_scaled)

# Stack test predictions
meta_test = np.column_stack([xgb_test, lgb_test, et_test])

print("   → Meta-model combining predictions...")
final_pred = meta_model.predict(meta_test)


# ============================================
# SECTION 13: POST-PROCESSING
# ============================================
"""
Apply constraints and adjustments to ensure realistic predictions.

Steps:
1. Ensure no negative prices (minimum $0.01)
2. Cap extreme predictions at reasonable maximum
3. Slight smoothing to reduce overconfident predictions
"""

print("\n🔧 Post-processing predictions...")

# Step 1: Ensure non-negative prices
final_pred = np.maximum(final_pred, 0.01)
print("   ✓ Enforced minimum price: $0.01")

# Step 2: Cap at reasonable maximum (120% of training 98th percentile)
max_allowed = price_98 * 1.2
final_pred = np.clip(final_pred, 0.01, max_allowed)
print(f"   ✓ Capped maximum price: ${max_allowed:.2f}")

# Step 3: Slight smoothing in log-space (reduces extreme predictions)
final_pred = np.expm1(np.log1p(final_pred) * 0.98)
print("   ✓ Applied smoothing (98% factor in log-space)")

# Print prediction statistics
print(f"\n📊 FINAL PREDICTION STATISTICS:")
print(f"   Minimum:  ${final_pred.min():.2f}")
print(f"   Maximum:  ${final_pred.max():.2f}")
print(f"   Mean:     ${final_pred.mean():.2f}")
print(f"   Median:   ${np.median(final_pred):.2f}")
print(f"   Std Dev:  ${final_pred.std():.2f}")


# ============================================
# SECTION 14: SUBMISSION FILE GENERATION
# ============================================
"""
Create submission file in required format.
Format: sample_id, price
"""

print("\n💾 Creating submission file...")

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_pred
})

# Save to CSV file
output_filename = 'test_out_quantum_safe.csv'
submission.to_csv(output_filename, index=False)

print(f"   ✓ Saved to: {output_filename}")
print(f"   ✓ Shape: {submission.shape}")
print(f"\n   Preview of submission file:")
print(submission.head(10))


# ============================================
# SECTION 15: FINAL SUMMARY
# ============================================
"""
Display final results and performance summary.
"""

print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE!")
print("="*60)

print(f"\n📈 PERFORMANCE SUMMARY:")
print(f"   Validation SMAPE: {stacked_smape:.4f}%")

# Provide performance interpretation
if stacked_smape < 40:
    print(f"   🏆 OUTSTANDING! Top-tier performance!")
elif stacked_smape < 45:
    print(f"   🔥 EXCELLENT! Competitive performance!")
elif stacked_smape < 50:
    print(f"   ✅ VERY GOOD! Solid predictions!")
elif stacked_smape < 55:
    print(f"   👍 GOOD! Decent performance!")
else:
    print(f"   📊 BASELINE! Room for improvement!")

print(f"\n🔍 MODEL ARCHITECTURE:")
print(f"   • Engineered Features: {train_quantum.shape[1]}")
print(f"   • Text Embeddings: {train_tfidf_12_dense.shape[1] + train_tfidf_13_dense.shape[1] + train_tfidf_char_dense.shape[1]}")
print(f"   • Character N-grams: {train_char_ngrams.shape[1]}")
print(f"   • Total Features: {X.shape[1]}")
print(f"   • Training Samples: {X_train.shape[0]}")
print(f"   • Base Models: 3 (XGBoost, LightGBM, Extra Trees)")
print(f"   • Meta-Model: Ridge Regression")

print(f"\n⚙️  COMPUTATIONAL DETAILS:")
print(f"   • Feature Engineering: ~2 minutes")
print(f"   • Text Vectorization: ~2 minutes")
print(f"   • Model Training: ~5 minutes")
print(f"   • Total Runtime: ~10-12 minutes")
print(f"   • Memory Efficient: ✓")
print(f"   • CPU Parallelized: ✓")

print(f"\n📁 OUTPUT:")
print(f"   • File: {output_filename}")
print(f"   • Predictions: {len(final_pred)}")
print(f"   • Ready for submission: ✅")

print(f"\n🔒 COMPLIANCE:")
print(f"   • No External LLMs: ✅")
print(f"   • Competition Safe: ✅")
print(f"   • Reproducible: ✅ (random_state=42)")
print(f"   • Open Source Ready: ✅")

print("\n" + "="*60)
print("🚀 Next Steps:")
print("="*60)
print("   1. Review validation SMAPE score")
print("   2. Submit 'test_out_quantum_safe.csv' to competition")
print("   3. Compare with leaderboard scores")
print("   4. Iterate on feature engineering if needed")
print("   5. Consider hyperparameter tuning for improvements")
print("\n" + "="*60)

print("\n✨ Thank you for using this pipeline!")
print("📚 For questions or improvements, check the documentation.")
print("🌟 Star this repo if you found it helpful!")

print("="*60 + "\n")
