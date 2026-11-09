# Work on a copy so the raw df stays unchanged
df_clean = df.copy()

# Median for skewed numeric columns
df_clean["annual_income"] = df_clean["annual_income"].fillna(df_clean["annual_income"].median())
df_clean["credit_score"] = df_clean["credit_score"].fillna(df_clean["credit_score"].median())

# Mode for the categorical column
df_clean["employment_type"] = df_clean["employment_type"].fillna(df_clean["employment_type"].mode()[0])

print("Missing values after cleaning:", df_clean.isna().sum().sum())
print("Rows kept:", len(df_clean))

# Why median: income is right-skewed with a few very high earners,
# so the mean is pulled up by outliers while the median reflects a typical applicant.
print("Income mean vs median:", round(df["annual_income"].mean()), round(df["annual_income"].median()))
