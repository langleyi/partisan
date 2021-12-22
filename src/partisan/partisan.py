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

            pd_dict[feat] = _pd_df.groupby(feat)[["pred"]].mean()

        return pd_dict

    def plot(self):
        pd_dict = self.values()
        # iterate through dictionary and plot
        col_nums = 3  # how many plots per row
        row_nums = math.ceil(len(pd_dict) / col_nums)  # how many rows of plots
        plt.figure(figsize=(10, 10))  # change the figure size as needed
        for i, (k, v) in enumerate(pd_dict.items(), 1):
            ax1 = plt.subplot(row_nums, col_nums, i)
            p = sns.lineplot(data=v, x=k, y="pred", ax=ax1)
            if is_numeric_dtype(self.X[k]):
                if self.X[k].nunique() > 10:
                    s = sns.histplot(
                        data=self.X[
                            (
                                self.X[k].between(
                                    self.X[k].quantile(0.05), self.X[k].quantile(0.95)
                                )
                            )
                            & (self.X[k].nunique() > 10)
                        ],
                        x=k,
                        alpha=0.01,
                        kde=False,
                        ax=ax1.twinx(),
                    )
                else:
                    s = sns.histplot(
                        data=self.X,
                        x=k,
                        alpha=0.1,
                        kde=False,
                        ax=ax1.twinx(),
                        color="k",
                    )

            plt.title(f"DataFrame: {k}")
            plt.ylabel("")
            plt.yticks([])

        plt.tight_layout()
        plt.show()


from sklearn.datasets import fetch_openml

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
X["y"] = y
X = X.dropna(subset={"pclass", "age", "sibsp", "parch", "fare"})
y = X["y"]
X = X[["pclass", "age", "sibsp", "parch", "fare"]]

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X, y)

exp = PDP(clf, X)
exp.plot()
