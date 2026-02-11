import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_df(df: pd.DataFrame, label_col: str, seed: int,
             train_ratio: float, val_ratio: float, test_ratio: float):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    y = df[label_col].astype(int)

    df_train, df_temp = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y
    )

    # split temp into val and test
    y_temp = df_temp[label_col].astype(int)
    val_share_of_temp = val_ratio / (val_ratio + test_ratio)

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=y_temp
    )

    return df_train, df_val, df_test


def main():
    ap = argparse.ArgumentParser(description="Create 70/15/15 splits from clean CSVs.")
    ap.add_argument("--static_csv", default="data/static_clean.csv")
    ap.add_argument("--dynamic_csv", default="data/dynamic_clean.csv")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    for name, path in [("static", args.static_csv), ("dynamic", args.dynamic_csv)]:
        df = pd.read_csv(path)

        if args.label_col not in df.columns:
            raise ValueError(f"{name}: missing '{args.label_col}' in {path}")

        # keep consistent with your existing dynamic cleaning
        df = df.drop(columns=["total_activity"], errors="ignore")

        df_train, df_val, df_test = split_df(
            df, label_col=args.label_col, seed=args.seed,
            train_ratio=args.train, val_ratio=args.val, test_ratio=args.test
        )

        df_train.to_csv(f"data/{name}_train.csv", index=False)
        df_val.to_csv(f"data/{name}_val.csv", index=False)
        df_test.to_csv(f"data/{name}_test.csv", index=False)

        print(f"\n{name.upper()} split sizes:")
        print(f"  train: {len(df_train)} -> data/{name}_train.csv")
        print(f"  val  : {len(df_val)} -> data/{name}_val.csv")
        print(f"  test : {len(df_test)} -> data/{name}_test.csv")


if __name__ == "__main__":
    main()
