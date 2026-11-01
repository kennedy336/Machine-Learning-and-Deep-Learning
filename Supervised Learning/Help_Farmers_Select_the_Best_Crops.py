import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

crops = pd.read_csv("soil_measures.csv")

X = crop_df.drop(columns=["crop"])
y = crop_df["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features_score = {}

for feature in X.columns:
    model = LogisticRegresion(max_iter=2000, multi_class="multinomial")
    model.fit(X_train[[feature]], y_train)
    y_pred = model.predict(X_test[[feature]])

    score = f1_score(y_test, y_pred, average="weighted")
    feature_score[feature] = score

best_feature_name = max(feature_score, key=feature_scores.get)
best_score = feature_scores[best_feature_name]

best_predictive_feature = {best_feature_name: best_score}
print(best_predictive_feature)