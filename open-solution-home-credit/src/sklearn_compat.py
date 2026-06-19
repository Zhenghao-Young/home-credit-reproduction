import joblib
import sklearn.externals as sklearn_externals

if not hasattr(sklearn_externals, "joblib"):
    sklearn_externals.joblib = joblib
