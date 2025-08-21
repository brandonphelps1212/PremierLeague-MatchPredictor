
"""
predict.py — Train & evaluate a Premier League match outcome model compatible with the tutorial CSV.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def _ensure_basic_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("CSV is missing a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "hour" not in df.columns:
        if "time" in df.columns:
            df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour
        else:
            df["hour"] = np.nan

    if "day_code" not in df.columns:
        if "day" in df.columns:
            day_map = {d:i for i,d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])}
            df["day_code"] = df["day"].map(day_map)
        else:
            df["day_code"] = df["date"].dt.dayofweek

    if "is_home" not in df.columns:
        if "venue" in df.columns:
            df["is_home"] = (df["venue"].str.lower() == "home").astype(int)
        else:
            df["is_home"] = 1
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_basic_cols(df)
    required = ["team", "opponent", "gf", "ga"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column '{c}'. Columns are: {list(df.columns)}")

    # Binary target: win (1) vs not-win (0)
    if "result" in df.columns:
        df["target"] = (df["result"].astype(str).str.upper().str.startswith("W")).astype(int)
    else:
        df["target"] = (df["gf"] > df["ga"]).astype(int)

    # Sort by team/date
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # Team prior form (exclude current match with shift)
    df["team_form_w5"] = (
        df.groupby("team")["target"]
          .rolling(5, min_periods=1).mean()
          .reset_index(level=0, drop=True)
    )
    df["team_form_w5"] = df.groupby("team")["team_form_w5"].shift(1)

    for w in (3, 5):
        for stat in ["gf", "ga"]:
            col = f"{stat}_ma{w}"
            df[col] = (
                df.groupby("team")[stat].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            df[col] = df.groupby("team")[col].shift(1)  # prior averages
        df[f"gd_ma{w}"] = df[f"gf_ma{w}"] - df[f"ga_ma{w}"]

    # Fill beginning NaNs with neutral priors
    for c in ["team_form_w5","gf_ma3","ga_ma3","gd_ma3","gf_ma5","ga_ma5","gd_ma5"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mean())

    return df

def train_eval(df: pd.DataFrame, test_start: str, model_name: str = "rf", tune: bool = False):
    df = engineer_features(df)
    df = df.sort_values("date")
    train = df[df["date"] < pd.to_datetime(test_start)]
    test = df[df["date"] >= pd.to_datetime(test_start)]

    X_cols_num = ["hour", "day_code", "is_home",
                  "team_form_w5",
                  "gf_ma3", "ga_ma3", "gd_ma3",
                  "gf_ma5", "ga_ma5", "gd_ma5"]
    X_cols_cat = ["team", "opponent"]
    y_col = "target"

    pre = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X_cols_cat),
            ("passthrough", "passthrough", X_cols_num),
        ]
    )

    if model_name == "gb":
        clf = GradientBoostingClassifier(random_state=42)
        param_grid = {"clf__n_estimators": [100, 200],
                      "clf__learning_rate": [0.05, 0.1],
                      "clf__max_depth": [2, 3]}
    else:
        clf = RandomForestClassifier(random_state=42, class_weight="balanced_subsample")
        param_grid = {"clf__n_estimators": [200, 400],
                      "clf__max_depth": [8, 12, None],
                      "clf__min_samples_split": [2, 5]}

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    if tune:
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(pipe, param_grid, cv=tscv, n_jobs=-1, scoring="accuracy", refit=True)
        gs.fit(train[X_cols_cat + X_cols_num], train[y_col])
        model = gs.best_estimator_
        best = gs.best_params_
    else:
        model = pipe.fit(train[X_cols_cat + X_cols_num], train[y_col])
        best = None

    def report(X, y_true):
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    train_metrics = report(train[X_cols_cat + X_cols_num], train[y_col])
    test_metrics = report(test[X_cols_cat + X_cols_num], test[y_col])

    return {
        "params": best,
        "train": train_metrics,
        "test": test_metrics,
        "n_train": len(train),
        "n_test": len(test),
        "columns_num": X_cols_num,
        "columns_cat": X_cols_cat,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="matches.csv")
    ap.add_argument("--test_start", type=str, default="2022-01-01")
    ap.add_argument("--model", type=str, choices=["rf", "gb"], default="rf")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = train_eval(df, test_start=args.test_start, model_name=args.model, tune=args.tune)
    import json
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
