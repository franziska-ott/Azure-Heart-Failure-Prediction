import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Scalar value that regulates the magnitude of parameter updates during training.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=3,
        help="Specifies the number of boosting stages to perform.",
    )

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Learning Rate:", np.float(args.learning_rate))
    run.log("Number Estimators:", np.int(args.n_estimators))

    ws = run.experiment.workspace
    
    key = "Heart-Failure-Dataset"
    dataset = ws.datasets[key]
    
    df = dataset.to_pandas_dataframe()

    y = df["DEATH_EVENT"]
    x = df.drop(columns=["DEATH_EVENT"])
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    model = GradientBoostingClassifier(learning_rate=args.learning_rate, n_estimators=args.n_estimators).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    model_filename = f"trained_model.pkl"
    joblib.dump(value=model, filename=model_filename)

    run.upload_file(model_filename, model_filename)


if __name__ == "__main__":
    main()
