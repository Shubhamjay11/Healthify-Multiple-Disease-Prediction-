import joblib
import pandas as pd
import model_dispatcher


def train_full(df_path, model, save_model=False):
    """
    Function to train the model on full train data
    :param df_path: path of train data
    :param model: model to train (key from model_dispatcher)
    :param save_model: to save model or not (True to save)
    """
    df = pd.read_csv(df_path)
    df.replace({"prognosis": config.disease_encode_dict}, inplace=True)
    X = df[config.symptoms]
    y = df["prognosis"]
    clf = model_dispatcher.MODELS[model]
    clf.fit(X, y)

    print("Model training done..")

    if save_model == "True":
        joblib.dump(clf, f"../../models/{model}_disease_pred.pkl")
        print("Model trained and saved successfully in models dir")


if __name__ == "__main__":
    import config
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--save_model", type=str)
    args = parser.parse_args()
    df_path = config.DISEASE_PREDICTION_PROCESSED
    train_full(df_path, args.model, args.save_model)


## To run this: python3 train_full_data.py --model <model_name> --save <bool>
