from sklearn.model_selection import train_test_split

data = df_clean.copy()

# New feature: loan size relative to income
data["loan_to_income"] = data["loan_amount"] / data["annual_income"]

# One-hot encode employment type
data = pd.get_dummies(data, columns=["employment_type"], drop_first=True, dtype=int)

# Features and target (drop ID and target from X)
X = data.drop(columns=["customer_id", "default"])
y = data["default"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("y_train:", y_train.shape, " y_test:", y_test.shape)
print("Features:", list(X.columns))
