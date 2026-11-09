from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# Scaled logistic regression with balanced class weights
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
model.fit(X_train, y_train)

# Test-set ROC-AUC
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Test ROC-AUC: {auc:.3f}")

print(classification_report(y_test, model.predict(X_test)))

# Top 5 features by absolute coefficient
coefs = pd.Series(model[-1].coef_[0], index=X_train.columns)
top5 = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(5)
print("Top 5 features:\n", top5.round(3))

print("Deployment view: AUC above 0.70 is good enough for an early-warning flag, not for automatic rejection. "
      "Missing a defaulter (false negative) costs the bank far more than reviewing a good applicant, "
      "so use a lower threshold to catch more defaulters and send flagged cases to manual review.")
