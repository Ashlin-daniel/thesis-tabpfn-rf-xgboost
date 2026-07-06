from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def train_regression(X_train, y_train, X_test, y_test):

    X_train = pd.DataFrame(X_train)
    X_test  = pd.DataFrame(X_test)

    y_train = pd.DataFrame(y_train).iloc[:, 0]
    y_test  = pd.DataFrame(y_test).iloc[:, 0]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred))
    }

def train_regression_tabpfn(X_train, y_train, X_test, y_test):

    import pandas as pd
    import numpy as np
    from tabpfn import TabPFNRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    X_train = pd.DataFrame(X_train)
    X_test  = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train).iloc[:, 0]
    y_test  = pd.DataFrame(y_test).iloc[:, 0]

    model = TabPFNRegressor(device="cpu")  # or "cuda"

    model.fit(X_train.values, y_train.values)
    pred = model.predict(X_test.values)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred))
    }