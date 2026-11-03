# ============================================================================
# FUNCTIONS:
# Read batch data, compare models, estimate uncertainty, and optimize profit.
# ============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from src.classes import MAbRegressor
from src.constants import SELECTED_BATCH_IDS

INITIAL = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
INPUTS = ["Initial Glucose [mM]", "Initial Glutamine [mM]", "Initial VCD [cells/mL]", "Harvest time [d]"]
TARGET = "mAb [mg/L]"


def load_batches(folder, conditions, selected_ids):
    """Read 30 independent batch time series; add constant INITIAL values to each time point."""
    lookup = conditions.set_index("Batch ID")
    if not lookup.index.is_unique:
        raise ValueError("Initial conditions contains duplicate batch IDs")
    missing_ids = set(selected_ids) - set(lookup.index)
    if missing_ids:
        raise ValueError(f"Batch IDs not in initial_conditions.csv: {sorted(missing_ids)}")
    missing_files = [f"batch_{b:04d}.csv" for b in selected_ids if not (folder / f"batch_{b:04d}.csv").exists()]
    if missing_files:
        raise FileNotFoundError("Download these 30 selected CSV files into datasets/:\n" + ", ".join(missing_files))
    frames = []
    for batch_id in selected_ids:
        run = pd.read_csv(folder / f"batch_{batch_id:04d}.csv")
        if "Time [h]" not in run or TARGET not in run:
            raise ValueError(f"Batch {batch_id}: missing Time [h] or mAb [mg/L]")
        run = run.sort_values("Time [h]").copy()
        if run["Time [h]"].duplicated().any():
            raise ValueError(f"Batch {batch_id}: duplicate time points")
        run["Batch ID"] = batch_id
        run["Initial Glucose [mM]"] = lookup.loc[batch_id, INITIAL[0]]
        run["Initial Glutamine [mM]"] = lookup.loc[batch_id, INITIAL[1]]
        run["Initial VCD [cells/mL]"] = lookup.loc[batch_id, INITIAL[2]]
        run["Harvest time [d]"] = run["Time [h]"] / 24.0
        frames.append(run)
    data = pd.concat(frames, ignore_index=True)
    if data[INPUTS + [TARGET]].isna().any().any():
        raise ValueError("Some input or output values are missing")
    return data


def model_specs():
    """Return model specifications."""
    return ([(f"Polynomial degree {d}", "polynomial", {"degree": d}) for d in (1, 2, 3)]
            + [(f"Time-constrained polynomial degree {d}", "rate_polynomial", {"degree": d}) for d in (1, 2)]
            + [
        (f"kNN k={k} {w}", "knn", {"neighbors": k, "weights": w})
        for k in (3, 5, 9) for w in ("uniform", "distance")])


def predict_metrics(actual, predicted):
    """Calculate RMSE (mg/L) and R²; lower RMSE is better."""
    return float(np.sqrt(mean_squared_error(actual, predicted))), float(r2_score(actual, predicted))


def compare_models(training):
    """Five-fold cross-validation, holding out entire batches in every fold."""
    splitter = GroupKFold(n_splits=5)
    x, y, groups = training[INPUTS], training[TARGET], training["Batch ID"]
    results = []
    for name, kind, params in model_specs():
        fold_rmse, fold_r2 = [], []
        for i_train, i_val in splitter.split(x, y, groups):
            model = MAbRegressor(kind, **params).fit(x.iloc[i_train], y.iloc[i_train])
            pred = model.predict(x.iloc[i_val])
            rmse, r2 = predict_metrics(y.iloc[i_val], pred)
            fold_rmse.append(rmse)
            fold_r2.append(r2)
        results.append({"Model": name, "CV RMSE mean [mg/L]": np.mean(fold_rmse),
                        "CV RMSE SD [mg/L]": np.std(fold_rmse, ddof=1),
                        "CV R2 mean": np.mean(fold_r2), "Kind": kind, "Parameters": str(params)})
    return pd.DataFrame(results).sort_values("CV RMSE mean [mg/L]").reset_index(drop=True)


def save_comparison_plot(scores, path):
    """Save a horizontal bar chart with validation variability across folds."""
    fig, ax = plt.subplots(figsize=(11, 7))
    s = scores.sort_values("CV RMSE mean [mg/L]", ascending=False)
    ax.barh(s["Model"], s["CV RMSE mean [mg/L]"], xerr=s["CV RMSE SD [mg/L]"], capsize=3)
    ax.set_xlabel("Grouped cross-validation RMSE [mg/L] (lower is better)")
    ax.set_title("Direct regression model comparison — 5 grouped folds")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_stage_errors(test, table_path, figure_path):
    """Compare RMSE in early, middle and late batch stages on unseen batches."""
    stages = [("Early (0-5 d)", 0, 5), ("Middle (5-15 d)", 5, 15),
              ("Late (15+ d)", 15, float("inf"))]
    records = []
    for label, low, high in stages:
        times = test[INPUTS[3]]
        mask = (times >= low) & (times < high)
        if mask.any():
            rmse, r2 = predict_metrics(test.loc[mask, TARGET],
                                       test.loc[mask, "Predicted mAb [mg/L]"])
            records.append({"Stage": label, "Test RMSE [mg/L]": rmse,
                            "Test R2": r2, "Number of observations": int(mask.sum())})
    stage_table = pd.DataFrame(records)
    stage_table.to_csv(table_path, index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(stage_table["Stage"], stage_table["Test RMSE [mg/L]"])
    ax.set(ylabel="Test RMSE [mg/L]", title="Held-out prediction error by culture stage")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    return stage_table


def batch_calibration_residuals(training, kind, params):
    """Out-of-fold mAb errors, keep maximum absolute error for each validation batch.

    These cross-validated errors are an approximate uncertainty estimate.
    """
    x, y, group = training[INPUTS], training[TARGET], training["Batch ID"]
    errors = []
    for train_idx, val_idx in GroupKFold(n_splits=5).split(x, y, group):
        model = MAbRegressor(kind, **params).fit(x.iloc[train_idx], y.iloc[train_idx])
        fold = training.iloc[val_idx][["Batch ID", TARGET]].copy()
        fold["Abs error"] = abs(fold[TARGET].to_numpy() - model.predict(x.iloc[val_idx]))
        errors.extend(fold.groupby("Batch ID")["Abs error"].max().tolist())
    # Numpy linear percentile gives an approximate 95th percentile of per-batch errors.
    return float(np.quantile(errors, 0.95))


def economics(frame):
    """Apply econ equations to predicted mAb at each harvest time."""
    out = frame.copy()
    out["Cell culture cost [$/L]"] = (10 * out[INPUTS[0]] + 20 * out[INPUTS[1]] + 0.0005 * out[INPUTS[2]])
    out["Cell culture value [$/L]"] = 1000 * out["Predicted mAb [mg/L]"]
    denominator = out[INPUTS[3]] + 1.0  # one day between batches
    out["Space-time profit [$/L/d]"] = (out["Cell culture value [$/L]"] - out["Cell culture cost [$/L]"]) / denominator
    out["Space-time yield [mg/L/d]"] = out["Predicted mAb [mg/L]"] / denominator
    return out


def optimize(model, conditions, selected, time_days):
    """Try the provided initial-condition grid and observed nonzero harvest times."""
    # The entire provided 1029-row table contains three replicates per unique condition.
    choices = conditions[INITIAL].drop_duplicates().reset_index(drop=True)
    times = np.sort(np.unique(time_days))
    times = times[times > 0]
    if len(times) == 0:
        raise ValueError("No positive harvest times were found")
    grid = choices.loc[choices.index.repeat(len(times))].reset_index(drop=True)
    grid.columns = INPUTS[:3]
    grid[INPUTS[3]] = np.tile(times, len(choices))
    grid["Predicted mAb [mg/L]"] = model.predict(grid[INPUTS])
    scores = economics(grid)
    winner = scores.loc[scores["Space-time profit [$/L/d]"].idxmax()].copy()
    return winner, scores


def add_intervals(winner, half_width):
    """Transform an approximate mAb error band into value/profit/yield bands.    """
    m = winner["Predicted mAb [mg/L]"]
    lower, upper = max(0.0, m - half_width), m + half_width
    t = winner[INPUTS[3]] + 1
    c = winner["Cell culture cost [$/L]"]
    winner["Approx mAb lower [mg/L]"] = lower
    winner["Approx mAb upper [mg/L]"] = upper
    winner["Approx value lower [$/L]"] = 1000 * lower
    winner["Approx value upper [$/L]"] = 1000 * upper
    winner["Approx profit lower [$/L/d]"] = (1000 * lower - c) / t
    winner["Approx profit upper [$/L/d]"] = (1000 * upper - c) / t
    winner["Approx yield lower [mg/L/d]"] = lower / t
    winner["Approx yield upper [mg/L/d]"] = upper / t
    return winner


def save_profit_plot(scores, winner, path):
    """Show the full profit-versus-time curve at the recommended starting conditions."""
    same = np.ones(len(scores), dtype=bool)
    for f in INPUTS[:3]:
        same &= np.isclose(scores[f].to_numpy(), winner[f])
    data = scores.loc[same].sort_values(INPUTS[3])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(data[INPUTS[3]], data["Space-time profit [$/L/d]"], marker="o", markersize=2)
    ax.scatter([winner[INPUTS[3]]], [winner["Space-time profit [$/L/d]"]], s=60, label="Recommended harvest")
    ax.set(xlabel="Harvest time [d]", ylabel="Predicted space-time profit [$/L/d]", title="Optimal initial conditions: effect of harvest time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_test_plot(test, path):
    """Plot predicted and measured mAb versus time on held-out test batches."""
    batches = list(test["Batch ID"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, batch_id in zip(axes.ravel(), batches):
        b = test[test["Batch ID"] == batch_id].sort_values(INPUTS[3])
        ax.plot(b[INPUTS[3]], b[TARGET], label="Measured")
        ax.plot(b[INPUTS[3]], b["Predicted mAb [mg/L]"], linestyle="--", label="Predicted")
        ax.set_title(f"Batch {batch_id:04d}")
        ax.set_xlabel("Days")
        ax.set_ylabel("mAb [mg/L]")
    axes.ravel()[0].legend()
    fig.suptitle("Held-out batch predictions (direct regression)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_project(root):
    """Run everything. main.py calls this ONE function."""
    root = Path(root)
    dataset_dir, table_dir, figure_dir = (root / s for s in ("datasets", "tables", "figures"))
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    conditions = pd.read_csv(dataset_dir / "initial_conditions.csv")
    selected = list(SELECTED_BATCH_IDS)
    if len(selected) != 30 or len(set(selected)) != 30:
        raise ValueError("SELECTED_BATCH_IDS must contain exactly 30 unique IDs")
    conditions.set_index("Batch ID").loc[selected].reset_index().to_csv(table_dir / "selected_batches.csv", index=False)
    data = load_batches(dataset_dir, conditions, selected)
    # 24 independent batches train/validate; 6 independent batches test.
    batch_ids = np.array(selected)
    train_i, test_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(batch_ids, groups=batch_ids))
    train_ids = set(batch_ids[train_i])
    test_ids = set(batch_ids[test_i])
    split_table = pd.DataFrame({"Batch ID": selected, "Dataset": ["Training" if b in train_ids else "Testing" for b in selected]})
    split_table.to_csv(table_dir / "train_test_batches.csv", index=False)
    training = data[data["Batch ID"].isin(train_ids)]
    testing = data[data["Batch ID"].isin(test_ids)].copy()
    scores = compare_models(training)
    scores.to_csv(table_dir / "model_comparison.csv", index=False)
    save_comparison_plot(scores, figure_dir / "model_comparison.png")
    best = scores.iloc[0]
    spec = next((kind, params) for name, kind, params in model_specs() if name == best["Model"])
    kind, params = spec
    tuned = MAbRegressor(kind, **params).fit(training[INPUTS], training[TARGET])
    testing["Predicted mAb [mg/L]"] = tuned.predict(testing[INPUTS])
    testing[["Batch ID", "Time [h]", TARGET, "Predicted mAb [mg/L]"]].to_csv(table_dir / "test_predictions.csv", index=False)
    save_test_plot(testing, figure_dir / "test_predictions.png")
    stage_table = save_stage_errors(testing, table_dir / "test_stage_errors.csv",
                                    figure_dir / "test_stage_errors.png")
    test_rmse, test_r2 = predict_metrics(testing[TARGET], testing["Predicted mAb [mg/L]"])
    pd.DataFrame([{"Selected model": best["Model"], "Test RMSE [mg/L]": test_rmse, "Test R2": test_r2,
                   "Training batches": len(train_ids), "Testing batches": len(test_ids)}]).to_csv(table_dir / "test_performance.csv", index=False)
    # Approximate uncertainty from out-of-fold residuals of the training batches.
    half_width = batch_calibration_residuals(training, kind, params)
    # Refit on all 30 experiments for the eventual optimization (test remains honest above).
    final_model = MAbRegressor(kind, **params).fit(data[INPUTS], data[TARGET])
    winner, candidate_scores = optimize(final_model, conditions, selected, data[INPUTS[3]].unique())
    winner = add_intervals(winner, half_width)
    winner.to_frame().T.to_csv(table_dir / "recommended_conditions.csv", index=False)
    save_profit_plot(candidate_scores, winner, figure_dir / "profit_vs_harvest_time.png")
    print(f"Best model: {best['Model']}")
    print(f"Held-out test RMSE: {test_rmse:.3f} mg/L; R²: {test_r2:.3f}")
    print("Held-out RMSE by stage:")
    for _, r in stage_table.iterrows():
        print(f"  {r['Stage']}: {r['Test RMSE [mg/L]']:.3f} mg/L")
    print(f"Recommended harvest: {winner[INPUTS[3]]:.2f} days")
    print(f"Estimated space-time profit: ${winner['Space-time profit [$/L/d]']:.2f}/L/day")
    print("Tables saved to tables/ and figures saved to figures/.")
