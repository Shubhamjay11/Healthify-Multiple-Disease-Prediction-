import pandas as pd
import joblib
from sklearn import metrics
import model_dispatcher


def train(fold, df, model, save_model=False):
    """
    Function to train the model fold-wise
    :param df: training data (pandas DataFrame)
    :param model: model to train (key from model_dispatcher)
    :param num_folds: number of folds to run
    :param save_model: to save model or not (True to save)
    """
    print(fold)
    print(config.FOLD_MAPPING.get(fold))
    train_df = df[df.kfold.isin(config.FOLD_MAPPING.get(fold))]
    valid_df = df[df.kfold == fold]

    train_df.replace({"prognosis": config.disease_encode_dict}, inplace=True)
    valid_df.replace({"prognosis": config.disease_encode_dict}, inplace=True)

    ytrain = train_df.prognosis.values
    yvalid = valid_df.prognosis.values

    train_df = train_df[config.symptoms]
    valid_df = valid_df[config.symptoms]

    clf = model_dispatcher.MODELS[model]
    clf.fit(train_df, ytrain)
    preds = clf.predict(valid_df)
    accuracy = metrics.accuracy_score(yvalid, preds)
    print(f"Algorithm: {model}, fold: {fold}, accuracy: {accuracy} ")

    if save_model == "True":
        joblib.dump(clf, f"../../models/{model}_{fold}_.pkl")

    return accuracy


if __name__ == "__main__":
    import config
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    parser.add_argument("--save", type=str)

    args = parser.parse_args()

    df = pd.read_csv(config.TRAIN_FOLDS)

    total_acc = 0
    for fold in range(args.fold):
        acc = train(fold=fold, df=df, model=args.model, save_model=args.save)
        total_acc += acc
    mean_acc = total_acc / args.fold
    print("=" * 40)
    print(f"Mean Accuracy: {mean_acc}")
