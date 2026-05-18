# =========================
# Random Forest Classifier
# Spaceship Titanic
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================
# 1. Load Data
# =========================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_passenger_id = test["PassengerId"]


# =========================
# 2. Preprocessing Function
# =========================

spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_cols = ["HomePlanet", "Destination", "CryoSleep", "VIP"]
drop_cols = ["PassengerId", "Name", "Cabin"]


def preprocess_data(df, is_train=True):
    df = df.copy()

    # 1) Economic consumption features: missing means not used, fill with 0
    for col in spending_cols:
        df[col] = df[col].fillna(0)

    # 2) Create TotalSpending before log transformation
    df["TotalSpending"] = df[spending_cols].sum(axis=1)

    # 3) Log1p transformation to reduce right skewness
    for col in spending_cols + ["TotalSpending"]:
        df[col] = np.log1p(df[col])

    # 4) Age: fill missing with median, then discretize into four ordered groups
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 60, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Drop original Age after creating AgeGroup
    df = df.drop(columns=["Age"])

    # 5) Categorical variables: mode imputation
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 6) Drop irrelevant features
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 7) Convert target variable
    if is_train:
        df["Transported"] = df["Transported"].astype(int)

    return df


train_processed = preprocess_data(train, is_train=True)
test_processed = preprocess_data(test, is_train=False)


# =========================
# 3. Split Features and Target
# =========================

X = train_processed.drop(columns=["Transported"])
y = train_processed["Transported"]

X_test = test_processed


# =========================
# 4. Train / Validation Split
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. Build Random Forest Pipeline
# =========================

cat_features = ["HomePlanet", "Destination", "CryoSleep", "VIP"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="passthrough"
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    criterion="gini"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_model)
    ]
)


# =========================
# 6. 5-Fold Stratified Cross Validation
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("5-Fold CV Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())


# =========================
# 7. Train Model and Validate
# =========================

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)

print("Validation Accuracy:", val_accuracy)


# =========================
# 8. Confusion Matrix
# =========================

cm = confusion_matrix(y_val, val_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Transported", "Transported"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()


# =========================
# 9. Feature Importance Plot
# =========================

# Get feature names after one-hot encoding
encoded_cat_features = model.named_steps["preprocessor"] \
    .named_transformers_["cat"] \
    .get_feature_names_out(cat_features)

numeric_features = [
    col for col in X.columns
    if col not in cat_features
]

feature_names = list(encoded_cat_features) + numeric_features

importances = model.named_steps["classifier"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.head(15))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df["Feature"].head(15)[::-1],
    feature_importance_df["Importance"].head(15)[::-1]
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Ranking Chart of Random Forest Model")
plt.tight_layout()
plt.show()


# =========================
# 10. Train on Full Dataset and Predict Test Set
# =========================

model.fit(X, y)

test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_passenger_id,
    "Transported": test_pred.astype(bool)
})

submission.to_csv("random_forest_submission.csv", index=False)

print("Submission file saved as: random_forest_submission.csv")
