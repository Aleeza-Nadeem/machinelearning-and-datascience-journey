# 💳 Credit Card Fraud Detection: End-to-End Machine Learning Pipeline

An end-to-end Machine Learning pipeline built to detect credit card fraud on highly imbalanced real-world transaction data (~1.29 million records). This project demonstrates data cleaning, feature engineering, baseline evaluation, ensemble modeling, regularization to prevent overfitting, and decision threshold tuning to balance business trade-offs.

---

## 📌 Executive Summary & Problem Framing

Credit card fraud causes massive financial losses for banks and severe frustration for customers. Building a model for fraud detection presents a unique challenge in applied machine learning:

* **The Problem:** Unauthorized transactions occurring across millions of daily operations.
* **The Victims:** 
  * **Banks:** Face direct monetary loss, chargeback fees, and operational costs.
  * **Customers:** Experience security stress, compromised accounts, and frozen cards due to false alarms.
* **The Culprits (Fraud Patterns):** Non-linear transaction behaviors (e.g., small testing charges followed by high-value transactions late at night in distant locations).
* **The Technical Barrier (Extreme Class Imbalance):** Fraudulent transactions account for **< 0.5%** of all data. Standard metrics like Accuracy are misleading—a naive model predicting "0% Fraud" achieves 99.5% accuracy while failing completely at stopping fraud.

---

## 🧹 Data Preprocessing & Feature Engineering

The dataset consists of **~1.29 million training rows** (`fraudTrain.csv`) and a separate unseen test set (`fraudTest.csv`).

### Preprocessing Steps:
1. **Feature Engineering:**
   * Parsed `trans_date_trans_time` to extract temporal features (`hour`).
   * Processed date of birth (`dob`) to calculate exact customer `age`.
   * Created behavioral aggregates: `card_trans_count` (user card frequency) and `amt_to_avg_ratio` (ratio of transaction amount relative to historic average spending).
2. **Feature Selection:** Dropped high-cardinality noise and identifiers (`first`, `last`, `street`, `trans_num`, `unix_time`, `cc_num`).
3. **Encoding & Transformations:**
   * One-Hot Encoded categorical attributes (`gender`, `category`, `state`).
   * Applied Log Transformation (`np.log1p`) to skewed numerical columns (`amt`, `city_pop`, `distance_km`).
   * Scaled numerical features using `StandardScaler`.
4. **Data Leakage Prevention:** Scalers and transformation parameters were fit **strictly on `X_train`** before being applied onto unseen test data.

---

## 🔬 Model Development & Iterative Pipeline

### Phase 1: Baseline Linear Models (Logistic Regression & Ridge)
* **Hypothesis:** Establish a fast linear benchmark.
* **Outcome:** Poor detection metrics.
* **Key Insight:** Fraud patterns are highly **non-linear**. A high transaction amount alone isn't fraud, but a high amount combined with 3 AM and a distant merchant *is*. Linear models cannot capture these multi-feature combinations effectively.

### Phase 2: Tree Ensembles (Random Forest)
* **Hypothesis:** Tree ensembles capture complex `IF-THEN` conditional rules using Bagging and Feature Randomness.
* **Addressing Overfitting:** Initial unconstrained trees memorized individual training rows (yielding a fake 100% score). To force broad generalization, strict regularization constraints were applied:
  * `max_depth = 6`
  * `min_samples_split = 20`
  * `min_samples_leaf = 10`

---

## 📈 Threshold Tuning & Business Impact

Classification models default to a `0.50` probability threshold. However, in fraud detection, the cost of a **False Positive** (blocking a real customer's card) versus a **False Negative** (missing a fraudulent charge) is asymmetric.

### Test Set Performance Across Thresholds:

| Metric / Threshold | 0.55 (Aggressive) | 0.70 (Optimal Balance) | 0.75 (Strict) |
| :--- | :--- | :--- | :--- |
| **Recall (Fraud Caught)** | **85%** | **~69%** | ~60% |
| **False Positives (False Alarms)** | 4,783 | **937** | **252** |
| **Precision** | 0.04 | **~0.15** | ~0.36 |
| **Business Trade-off** | Too many false alarms; frustrates customers | **Optimal operational balance** | High precision, but misses more fraud |

> **Final System Choice:** A probability threshold of **`0.70`** was selected, cutting false alarms by **over 80%** while retaining strong fraud capture rates.

--