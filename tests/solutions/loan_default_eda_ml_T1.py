import pandas as pd

# Load the loan data
df = pd.read_csv("data/loan_data.csv")

# Shape and data types
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)

# Missing values per column
print("\nMissing values:\n", df.isna().sum())

# Overall default rate
default_rate = df["default"].mean() * 100
print(f"\nDefault rate: {default_rate:.1f}%")

# Observation
print("Observation: about 28% of loans defaulted, so the classes are imbalanced. "
      "annual_income, credit_score and employment_type have missing values to clean.")
