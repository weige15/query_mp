#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze H1 sensitivity rows.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--high_bit", type=int, default=8)
    parser.add_argument("--sensitive_quantile", type=float, default=0.9)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def auroc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score)
    pos = y == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def average_precision(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / y.sum())


def one_hot(series, prefix):
    dummies = pd.get_dummies(series.astype(str), prefix=prefix, dtype=float)
    return dummies


def make_matrix(df, mode, train_columns=None, means=None, stds=None):
    numeric = pd.DataFrame(index=df.index)
    numeric["bit"] = df["bit"].astype(float)
    numeric["token_pos"] = df["token_pos"].astype(float)
    numeric["activation_norm"] = np.log1p(df["activation_norm"].astype(float))
    numeric["activation_outlier"] = df["activation_outlier"].astype(float)
    numeric["high_entropy"] = df["high_entropy"].astype(float)
    numeric["high_margin"] = df["high_margin"].astype(float)
    numeric["high_loss"] = df["high_loss"].astype(float)

    if mode == "layer_only":
        x = one_hot(df["module_name"], "module")
    else:
        x = pd.concat(
            [
                numeric,
                one_hot(df["module_name"], "module"),
                one_hot(df["category"], "category"),
            ],
            axis=1,
        )

    if train_columns is None:
        train_columns = list(x.columns)
    x = x.reindex(columns=train_columns, fill_value=0.0)

    numeric_cols = [c for c in numeric.columns if c in x.columns]
    if means is None:
        means = x[numeric_cols].mean()
        stds = x[numeric_cols].std().replace(0, 1.0)
    x.loc[:, numeric_cols] = (x[numeric_cols] - means) / stds
    return x.to_numpy(dtype=float), train_columns, means, stds


def train_logreg(x, y, steps, lr, l2):
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        p = sigmoid(x @ w)
        grad = (x.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logreg(w, x):
    return sigmoid(np.c_[np.ones(len(x)), x] @ w)


def calibration_table(y, score, bins=10):
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "score_lo": lo,
                "score_hi": hi,
                "count": int(mask.sum()),
                "avg_score": float(score[mask].mean()),
                "event_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(df, train_df, test_df, y_train, y_test, mode, args):
    x_train, cols, means, stds = make_matrix(train_df, mode)
    x_test, _, _, _ = make_matrix(test_df, mode, cols, means, stds)
    w = train_logreg(x_train, y_train, args.steps, args.lr, args.l2)
    train_score = predict_logreg(w, x_train)
    test_score = predict_logreg(w, x_test)

    x_all, _, _, _ = make_matrix(df, mode, cols, means, stds)
    all_score = predict_logreg(w, x_all)

    metrics = {
        "mode": mode,
        "train_auroc": auroc(y_train, train_score),
        "test_auroc": auroc(y_test, test_score),
        "train_average_precision": average_precision(y_train, train_score),
        "test_average_precision": average_precision(y_test, test_score),
    }
    feature_weights = pd.DataFrame({"feature": ["intercept"] + cols, "weight": w})
    return metrics, all_score, test_score, feature_weights


def routing_curve(df, score, high_bit):
    rows = []
    for bit, bit_df in df.groupby("bit"):
        scores = score[bit_df.index.to_numpy()]
        deltas = bit_df["loss_delta"].clip(lower=0).to_numpy()
        for threshold in np.linspace(0, 1, 21):
            route_high = scores >= threshold
            rows.append(
                {
                    "policy": "dynamic_predictor",
                    "low_bit": int(bit),
                    "threshold": float(threshold),
                    "avg_bit": float(np.where(route_high, high_bit, bit).mean()),
                    "high_fraction": float(route_high.mean()),
                    "proxy_loss_delta": float(np.where(route_high, 0.0, deltas).mean()),
                }
            )
        for high_fraction in np.linspace(0, 1, 21):
            cutoff = np.quantile(deltas, 1 - high_fraction) if high_fraction > 0 else float("inf")
            route_high = deltas >= cutoff
            rows.append(
                {
                    "policy": "oracle_by_measured_delta",
                    "low_bit": int(bit),
                    "threshold": float(high_fraction),
                    "avg_bit": float(np.where(route_high, high_bit, bit).mean()),
                    "high_fraction": float(route_high.mean()),
                    "proxy_loss_delta": float(np.where(route_high, 0.0, deltas).mean()),
                }
            )
        rows.append(
            {
                "policy": "uniform_low",
                "low_bit": int(bit),
                "threshold": 0.0,
                "avg_bit": float(bit),
                "high_fraction": 0.0,
                "proxy_loss_delta": float(deltas.mean()),
            }
        )
        rows.append(
            {
                "policy": "uniform_high",
                "low_bit": int(bit),
                "threshold": 1.0,
                "avg_bit": float(high_bit),
                "high_fraction": 1.0,
                "proxy_loss_delta": 0.0,
            }
        )
    return pd.DataFrame(rows)


def top_table(df, group_cols, metric="loss_delta", n=20):
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_loss_delta=(metric, "mean"),
            p90_loss_delta=(metric, lambda x: x.quantile(0.9)),
            top1_flip_rate=("top1_flip", "mean"),
            rows=(metric, "size"),
        )
        .reset_index()
        .sort_values(["mean_loss_delta", "p90_loss_delta"], ascending=False)
        .head(n)
    )


def markdown_table(df):
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_summary(out_dir, df, threshold, metrics, routing):
    module_top = top_table(df, ["bit", "module_name"]).head(10)
    layer_top = top_table(df, ["bit", "layer_id"]).head(10)
    category_top = top_table(df, ["bit", "category"]).head(10)
    pos = df.assign(pos_bucket=(df["token_pos"] // 32) * 32)
    pos_top = top_table(pos, ["bit", "pos_bucket"]).head(10)
    usable = routing[routing["policy"] == "dynamic_predictor"].copy()
    best = usable.sort_values(["proxy_loss_delta", "avg_bit"]).head(10)

    lines = [
        "# H1 Sensitivity Analysis",
        "",
        f"Rows: {len(df)}",
        f"Sensitive label: loss_delta >= {threshold:.6g}",
        "",
        "## Predictor",
        "",
    ]
    for item in metrics:
        lines.append(
            f"- {item['mode']}: test AUROC={item['test_auroc']:.4f}, "
            f"test AP={item['test_average_precision']:.4f}"
        )
    lines += [
        "",
        "## Most Sensitive Modules",
        "",
        markdown_table(module_top),
        "",
        "## Most Sensitive Layers",
        "",
        markdown_table(layer_top),
        "",
        "## Query Categories",
        "",
        markdown_table(category_top),
        "",
        "## Token Position Buckets",
        "",
        markdown_table(pos_top),
        "",
        "## Best Dynamic Routing Proxy Points",
        "",
        markdown_table(best),
        "",
        "Note: routing rows are proxy estimates from one-module perturbations, not a full routed-model PPL run.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df["loss_delta"] = df["loss_delta"].astype(float)
    threshold = float(df["loss_delta"].quantile(args.sensitive_quantile))
    df["high_sensitivity"] = (df["loss_delta"] >= threshold).astype(int)

    sample_ids = np.array(sorted(df["sample_id"].unique()))
    split = max(1, int(round(len(sample_ids) * (1 - args.test_fraction))))
    train_ids = set(sample_ids[:split])
    train_df = df[df["sample_id"].isin(train_ids)]
    test_df = df[~df["sample_id"].isin(train_ids)]
    if test_df.empty:
        test_df = train_df

    y_train = train_df["high_sensitivity"].to_numpy(dtype=float)
    y_test = test_df["high_sensitivity"].to_numpy(dtype=float)

    metrics = []
    layer_metrics, layer_score, _, layer_weights = evaluate_model(
        df, train_df, test_df, y_train, y_test, "layer_only", args
    )
    dynamic_metrics, dynamic_score, test_score, dynamic_weights = evaluate_model(
        df, train_df, test_df, y_train, y_test, "dynamic", args
    )
    metrics.extend([layer_metrics, dynamic_metrics])

    out_metrics = {
        "sensitive_threshold": threshold,
        "positive_rate": float(df["high_sensitivity"].mean()),
        "train_samples": [int(x) for x in sorted(train_ids)],
        "test_samples": [int(x) for x in sorted(set(sample_ids) - train_ids)],
        "metrics": metrics,
    }
    (out_dir / "predictor_metrics.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")

    layer_weights.reindex(layer_weights["weight"].abs().sort_values(ascending=False).index).to_csv(
        out_dir / "layer_only_feature_importance.csv", index=False
    )
    dynamic_weights.reindex(dynamic_weights["weight"].abs().sort_values(ascending=False).index).to_csv(
        out_dir / "dynamic_feature_importance.csv", index=False
    )
    calibration_table(y_test, test_score).to_csv(out_dir / "calibration.csv", index=False)

    scored = df.copy()
    scored["layer_only_score"] = layer_score
    scored["dynamic_score"] = dynamic_score
    scored.to_csv(out_dir / "sensitivity_scored.csv", index=False)

    routing = routing_curve(scored, scored["dynamic_score"].to_numpy(), args.high_bit)
    routing.to_csv(out_dir / "routing_proxy.csv", index=False)

    top_table(scored, ["bit", "module_name"]).to_csv(out_dir / "sensitive_modules.csv", index=False)
    top_table(scored, ["bit", "layer_id"]).to_csv(out_dir / "sensitive_layers.csv", index=False)
    top_table(scored, ["bit", "category"]).to_csv(out_dir / "sensitive_categories.csv", index=False)
    top_table(scored.assign(pos_bucket=(scored["token_pos"] // 32) * 32), ["bit", "pos_bucket"]).to_csv(
        out_dir / "sensitive_token_positions.csv", index=False
    )

    write_summary(out_dir, scored, threshold, metrics, routing)


if __name__ == "__main__":
    main()
