if __name__ == "__main__":
    import config
    import pandas as pd
    from sklearn import model_selection

    df = pd.read_csv(config.DISEASE_PREDICTION_PROCESSED)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    y = df.prognosis.values

    kf = model_selection.StratifiedKFold(
        n_splits=config.num_folds, shuffle=True, random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, "kfold"] = fold

    df.to_csv(config.TRAIN_FOLDS, index=False)
