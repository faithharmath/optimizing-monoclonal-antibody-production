# ============================================================
# Direct regression models for mAb concentration
# ============================================================
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


class MAbRegressor:

    def __init__(self, kind="polynomial", degree=2, neighbors=5, weights="distance"):
        self.kind = kind
        if kind in ("polynomial", "rate_polynomial"):
            self.pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                ("regression", LinearRegression()),
            ])
        elif kind == "knn":
            self.pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("regression", KNeighborsRegressor(n_neighbors=neighbors, weights=weights)),
            ])
        else:
            raise ValueError("kind must be 'polynomial', 'rate_polynomial', or 'knn'")

    def fit(self, x, y):
        # Train model--> rate models train on mAb/t at only positive times.
        times = x["Harvest time [d]"].to_numpy(dtype=float)
        if self.kind == "rate_polynomial":
            keep = times > 0
            if not np.any(keep):
                raise ValueError("Rate model requires positive harvest times")
            self.pipeline.fit(x.loc[keep], np.asarray(y, dtype=float)[keep] / times[keep])
        else:
            self.pipeline.fit(x, y)
        return self

    def predict(self, x):
        # Predict mAb concentration while keeping mAb(0) = 0

        import numpy as np

        # Predict mAb concentration using the trained model
        predictions = self.pipeline.predict(x)

        # Prevent negative concentrations
        predictions = np.maximum(predictions, 0.0)

        # Get the harvest time for each prediction
        time = x["Harvest time [d]"].to_numpy()

        # At time zero, mAb concentration must equal zero
        predictions[np.isclose(time, 0.0)] = 0.0

        return predictions

