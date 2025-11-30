import ast
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def _to_list(self, x):
        # identical logic to what we discussed before
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            # string that looks like a Python list: "['Drama', 'Crime']"
            try:
                return ast.literal_eval(x)
            except Exception:
                pass
        if isinstance(x, str) and "," in x:
            # "Drama, Crime" → ["Drama", "Crime"]
            return [v.strip() for v in x.split(",") if v.strip()]
        if isinstance(x, str):
            # single label: "Drama"
            return [x.strip()]
        return []

    def fit(self, X, y=None):
        # X will be (n_samples, 1) from ColumnTransformer
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X[:, 0])
        data = col.apply(self._to_list).tolist()
        self.mlb.fit(data)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X[:, 0])
        data = col.apply(self._to_list).tolist()
        return self.mlb.transform(data)
