
# Sentiment Analysis Project Report

This report summarizes the sentiment analysis project, detailing the dataset, preprocessing steps, machine learning models employed, and key findings.

## 1. Dataset Overview

*   **Name:** Call-Center-Sentiment-Sample-Data
*   **Source:** Local upload (`/content/Call-Center-Sentiment-Sample-Data (2).xlsx`)
*   **Original Features:**
    *   `ID`: Unique identifier
    *   `Customer Name`: Name of the customer
    *   `Sentiment`: Original 5-class sentiment label (Very Negative, Negative, Neutral, Positive, Very Positive) - *Target variable*
    *   `CSAT Score`: Customer Satisfaction Score
    *   `Call Timestamp`: Timestamp of the call
    *   `Reason`: Textual reason for the call - *Primary feature for sentiment analysis*
    *   `City`: City of the customer
    *   `State`: State of the customer
    *   `Channel`: Communication channel
    *   `Response Time`: Response time category
    *   `Call Duration (Minutes)`: Duration of the call
    *   `Call Center`: Associated call center
*   **Target Variable:** `Sentiment`

## 2. Data Preprocessing

1.  **Data Loading:** The Excel file was loaded, handling header identification issues by programmatically finding the header row.
2.  **Missing Values:** Columns with all missing values were dropped.
3.  **Data Type Conversion:**
    *   `CSAT Score`, `Call Duration (Minutes)`: Converted to numeric.
    *   `Call Timestamp`: Converted to datetime.
4.  **Feature Selection:** `Reason` column was used as the primary text feature (X), and `Sentiment` as the target variable (y).
5.  **Initial Preprocessing (TF-IDF):** Text data was converted into numerical features using `TfidfVectorizer`.
6.  **Class Imbalance (Initial Attempt & Correction):**
    *   The dataset exhibited significant class imbalance (70 entries across 5 classes).
    *   An initial attempt with SMOTE on the full dataset led to data leakage and poor results.
    *   **Corrected Approach:** `ImbalancedPipeline` from `imblearn` was used, integrating `TfidfVectorizer`, `SMOTE` (applied only to training folds), and `LogisticRegression` to prevent data leakage.
7.  **Sentiment Class Simplification:** Due to the small dataset size and persistent low performance on minority classes, the 5 original sentiment classes were simplified to 3:
    *   'Very Negative' merged into 'Negative'.
    *   'Very Positive' merged into 'Positive'.
    *   'Neutral' remained.

## 3. Models and Training

Several machine learning models were trained and evaluated:

1.  **Initial Models (with TF-IDF, no advanced preprocessing/SMOTE):**
    *   **Logistic Regression:** Achieved ~42.86% accuracy.
    *   **Multinomial Naive Bayes:** Achieved ~35.71% accuracy.
    *   **Linear SVC:** Achieved ~42.86% accuracy.
    *   **Decision Tree:** Achieved ~42.86% accuracy.
    *   **Random Forest:** Achieved ~42.86% accuracy.
    *   *Observation:* All initial models showed poor performance for minority classes, indicating severe class imbalance.

2.  **Hyperparameter Tuning with GridSearchCV:**
    *   A `Pipeline` combining `TfidfVectorizer` and `LogisticRegression` was used.
    *   `GridSearchCV` was employed to tune `max_features`, `ngram_range` for TF-IDF, and `C`, `solver` for Logistic Regression.
    *   Best cross-validation accuracy: ~41.06%. Test set accuracy: ~42.86%.
    *   Still, minority classes had 0.00 precision/recall.

3.  **Model Training with ImbalancedPipeline (5-class, SMOTE in pipeline):**
    *   An `ImbalancedPipeline` with `TfidfVectorizer`, `SMOTE`, and `LogisticRegression` was used.
    *   `GridSearchCV` tuned parameters for all stages within the pipeline.
    *   Best cross-validation accuracy: ~12.48%. Test set accuracy: ~7.14%.
    *   *Conclusion:* SMOTE was ineffective with the 5-class problem due to extreme data scarcity, resulting in even worse performance.

4.  **Model Training with ImbalancedPipeline (Simplified 3-class, SMOTE in pipeline):**
    *   The target variable was simplified to 3 classes (Negative, Neutral, Positive).
    *   An `ImbalancedPipeline` with `TfidfVectorizer`, `SMOTE`, and `LogisticRegression` was re-trained.
    *   `GridSearchCV` was applied to this simplified problem.
    *   **Best Cross-Validation Accuracy:** ~46.59%
    *   **Test Set Accuracy:** ~42.86%
    *   **Classification Report Summary (Simplified):**
        *   `Negative`: Precision ~0.56, Recall ~0.56, F1-score ~0.56
        *   `Neutral`: Precision ~0.00, Recall ~0.00, F1-score ~0.00
        *   `Positive`: Precision ~0.33, Recall ~0.33, F1-score ~0.33
    *   *Finding:* While accuracy improved slightly, the 'Neutral' class still performed very poorly, indicating the challenges of a very small dataset even after simplification and balancing techniques.

## 4. Final Model

The `best_simplified_sentiment_model.joblib` represents the Logistic Regression model trained using `ImbalancedPipeline` on the 3-class simplified sentiment problem, with optimized hyperparameters and SMOTE applied within the cross-validation folds.

## 5. Potential Improvements

*   **More Data:** The most crucial improvement would be to acquire a larger dataset.
*   **Advanced Text Preprocessing:** Experiment with stemming, more refined stop word lists, and handling negations.
*   **Word Embeddings:** Explore pre-trained word embeddings (e.g., Word2Vec, GloVe) or contextual embeddings (e.g., BERT) for richer text representations.
*   **Different Models:** Investigate other models suitable for small, imbalanced text datasets or transfer learning approaches.
*   **Custom Loss Functions/Metrics:** For imbalanced data, consider optimizing for metrics like F1-score for specific classes, or using custom loss functions.
