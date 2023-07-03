import argparse
import numpy as np
import sklearn.ensemble as GradientBoostingClassifier
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(df):
    # Dict for cleaning data
    # Rename columns
    df = df.rename(columns=lambda x: x.lower().replace(' ', '_'))

    bmi = {
        "Normal": 0,
        "Overweight": 1,
        "Normal Weight": 2,
        "Obese": 3
    }

    sleep_disorder = {
        "None": 0,
        "Sleep Apnea": 1,
        "Insomnia": 2
    }

    # Clean and one hot encode data
    x_df = df.to_pandas_dataframe().dropna()
    x_df.drop("person_id", inplace=True, axis=1)
    x_df["gender"] = x_df.gender.apply(lambda s: 1 if s == "Male" else 0)
    occupation = pd.get_dummies(x_df.occupation, prefix="occupation")
    x_df.drop("occupation", inplace=True, axis=1)
    x_df = x_df.join(occupation)
    x_df["bmi_category"] = x_df.bmi_category.map(bmi)
    blood_pressure = pd.get_dummies(x_df.blood_pressure, prefix="blood_pressure")
    x_df.drop("blood_pressure", inplace=True, axis=1)
    x_df = x_df.join(blood_pressure)

    y_df = x_df.pop("sleep_disorder").map(sleep_disorder)
    
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Scalar value that regulates the magnitude of parameter updates during training.",
    )
    #[0.01, 0.05, 0.1, 0.2, 0.3]
    parser.add_argument(
        "--max_depth",
        type=int,
        default=100,
        help="Specifies the maximum depth allowed for an individual decision tree in the ensemble.",
    )
    #[1, 3, 5, 7, 9]
    args = parser.parse_args()

    run = Run.get_context()

    run.log("Learning Rate:", np.float(args.learning_rate))
    run.log("Max Depth:", np.int(args.max_depth))

    url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    ds = TabularDatasetFactory.from_delimited_files(url)

    x, y = clean_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    model = GradientBoostingClassifier(learning_rate=args.learning_rate, max_depth=args.max_depth).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    model_filename = f"model_{args.max_depth}_{args.learning_rate}.pkl"
    joblib.dump(value=model, filename=model_filename)

    run.upload_file(model_filename, model_filename)


if __name__ == "__main__":
    main()
