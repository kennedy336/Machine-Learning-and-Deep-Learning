

#Visualizing regresion models performance using boxplots
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
  
  results.append(cv_scores)
plt.boxplot(results, labels=modAels.keys())
plt.show()


#Predicting on the test set and calculating RMSE for each model
from sklearn.metrics import root_mean_squared_error

for name, model in models.items():
  model.fit(X_train_scaled, y_train)
  
  y_pred = model.predict(X_test_scaled)
  
  test_rmse = root_mean_squared_error(y_test, y_pred)
  print("{} Test Set RMSE: {}".format(name, test_rmse))
  
#visualizing classification models performance using boxplots
models = {
    "Logistic Regression": LogisticRegression(), 
    "KNN": KNeighborsClassifier(), 
    "Decision Tree Classifier": DecisionTreeClassifier()
}
results = []

for model in models.values():
  
  kf = KFold(n_splits=6, shuffle=True, random_state=12)
  
  # Perform cross-validation
  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
  results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.show()

#Pipeline for classification models
# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))