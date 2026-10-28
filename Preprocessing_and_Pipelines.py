# Create music_dummies
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)

# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies))

X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))