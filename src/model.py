from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class IrisModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)
        self.df = pd.read_csv("src/iris.csv")
        self.label_encoder = LabelEncoder()

    def train(self) -> None:
        # Split X and y correctly
        X = self.df.iloc[:, :-1].values
        y = self.label_encoder.fit_transform(self.df.iloc[:, -1])

        self.model.fit(X, y)

    def predict(self, features: list | np.ndarray) -> str:
        """
        features: [sepal_length, sepal_width, petal_length, petal_width]
        """
        features = np.array(features).reshape(1, -1)
        pred = self.model.predict(features)[0]
        return self.label_encoder.inverse_transform([pred])[0]


def train_and_save_model(model_path: str | Path = "data.pkl") -> Path:
    cls = IrisModel()
    cls.train()
    model_path = Path(model_path)
    joblib.dump(cls, model_path)
    return model_path
