import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical_dtype
import seaborn as sns
import math


class PDP:
    "Partial dependence plot class"

    def __init__(self, model, X: pd.DataFrame):
        """
        Args:
            model (sklearn model): Trained scikit-learn model. Must have either a `predict` method
                (if a regression model) or `predict_proba` method (if a classifier).
            X (pd.DataFrame): Pandas dataframe of regressors from which to generate the PDP plot.
        """
        self.model = model
        self.X = X

    def values(self):
        """ """
        pd_dict = {}
        for feat in self.X.columns:
            _feat_df = self.X[[feat]].copy()
            _oth_df = self.X[[i for i in self.X.columns if i != feat]].copy()

            _feat_df["helper"] = 1
            _oth_df["helper"] = 1
            _pd_df = _feat_df.merge(_oth_df, how="left", on="helper").drop(
                columns={"helper"}
            )

            if hasattr(self.model, "predict_proba"):
                _pd_df["pred"] = self.model.predict_proba(_pd_df)[:, 1]
            else:
                _pd_df["pred"] = self.model.predict(_pd_df)

            if is_numeric_dtype(_pd_df[feat]):
                if _pd_df[feat].nunique() > 10:
                    _pd_df = _pd_df[
                        _pd_df[feat].between(
                            _pd_df[feat].quantile(0.05), _pd_df[feat].quantile(0.95)
                        )
                    ]
            pd_dict[feat] = (
                _pd_df.groupby(feat)[["pred"]].mean().rename(columns={"pred": "PD"})
            )

        return pd_dict

    def plot(self):
        pd_dict = self.values()
        # iterate through dictionary and plot
        col_nums = 1  # how many plots per row
        row_nums = math.ceil(len(pd_dict) / col_nums)  # how many rows of plots
        plt.figure(
            figsize=(5, np.floor((row_nums * 5) / 2.5))
        )  # change the figure size as needed
        plt.rcParams.update({"font.sans-serif": "Arial"})
        for i, (k, v) in enumerate(pd_dict.items(), 1):
            if is_numeric_dtype(self.X[k]):
                ax1 = plt.subplot(row_nums, col_nums, i)
                lp = sns.lineplot(
                    data=v,
                    x=k,
                    y="PD",
                    ax=ax1,
                    color="tab:red",
                )
                lp.set(xlabel=None)

                if self.X[k].nunique() > 10:
                    data = self.X[
                        self.X[k].between(
                            self.X[k].quantile(0.05), self.X[k].quantile(0.95)
                        )
                    ]
                else:
                    data = self.X

                hp = sns.histplot(
                    data=data,
                    x=k,
                    alpha=0.1,
                    kde=False,
                    ax=ax1.twinx(),
                    color="k",
                )
                hp.set_ylabel(None)
                hp.set_xticks([])

            elif is_string_dtype(self.X[k]) | is_categorical_dtype(self.X[k]):
                top_n_vals = self.X[k].value_counts().head(10).index
                data = v[v.index.isin(top_n_vals)]
                ax1 = plt.subplot(row_nums, col_nums, i)
                bp = sns.barplot(
                    data=data.reset_index(),
                    x="PD",
                    y=k,
                    color="tab:red",
                    alpha=0.3,
                    orient="h",
                    ax=ax1,
                )
                bp.set_ylabel(None)

            plt.title(f"{k}")
            # plt.yticks([])

        plt.tight_layout()
        plt.show()


# # ALL NUMERICAL CLASSIFICATION
# from sklearn.datasets import fetch_openml

# X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
# X["y"] = y
# X = X.dropna(subset={"pclass", "age", "sibsp", "parch", "fare"})
# y = X["y"]
# X = X[["pclass", "age", "sibsp", "parch", "fare"]]

# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(random_state=0).fit(X, y)
# exp = PDP(clf, X)
# exp.plot()

# ALL NUMERICAL REGRESSION

# from sklearn.datasets import fetch_openml

# X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
# X = X.dropna(subset={"pclass", "age", "sibsp", "parch", "fare"})
# y = X["age"]
# X = X[["pclass", "sibsp", "parch", "fare"]]
# from sklearn.linear_model import LinearRegression

# regr = LinearRegression().fit(X, y)
# exp = PDP(regr, X)
# exp.plot()

# MIXED NUMERICAL + CATEGORICAL CLASSIFICATION
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

clf.fit(X, y)
exp = PDP(clf, X)
exp.plot()
