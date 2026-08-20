# Titanic Survival Prediction

A machine learning model that predicts whether a passenger survived the Titanic
disaster, built with Python and scikit-learn. This project covers the full ML
pipeline: loading data, cleaning it, training a classifier, and evaluating it honestly.

## Results

| Metric | Score |
|--------|-------|
| Training accuracy | 82.6% |
| **Test accuracy** | **81.0%** |

The small gap between training and test accuracy (~1.6 points) indicates the model
generalizes well and is not overfitting — it learned the underlying pattern rather
than memorizing the training data.

## Approach

1. **Load** the Titanic dataset (via seaborn's built-in dataset).
2. **Clean the data:**
   - Dropped redundant and leakage-prone columns (e.g. `alive`, which directly
     encodes the target).
   - Filled missing `age` values with the column mean.
   - Encoded the categorical `sex` column as numeric (female = 1, male = 0).
3. **Split** the data into 80% training / 20% test sets.
4. **Train** a logistic regression classifier.
5. **Evaluate** using accuracy on both the training and held-out test sets.

## Key concepts demonstrated

- Handling missing values and categorical encoding
- Avoiding data leakage (removing columns that reveal the answer)
- Train/test split for honest evaluation
- Recognizing overfitting via the train/test accuracy gap

## Tech stack

- Python 3.12
- pandas
- scikit-learn
- seaborn

## Running it

```bash
source .venv/bin/activate
python titanic_survival.py
```

## Possible improvements

- Feature engineering (e.g. combining `sibsp` + `parch` into a `family_size` feature)
- Trying other models (Random Forest, gradient boosting)
- Hyperparameter tuning
- More sophisticated missing-value imputation (e.g. mean age per passenger class)