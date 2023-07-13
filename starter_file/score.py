# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Person ID": pd.Series([1], dtype="int64"), "Gender": pd.Series(["Female"], dtype="object"), "Age": pd.Series([20], dtype="int64"), "Occupation": pd.Series(["Doctor"], dtype="object"), "Sleep Duration": pd.Series([6.2], dtype="float64"), "Quality of Sleep": pd.Series([4], dtype="int64"), "Physical Activity Level": pd.Series([30], dtype="int64"), "Stress Level": pd.Series([6], dtype="int64"), "BMI Category": pd.Series(["Normal"], dtype="object"), "Blood Pressure": pd.Series(["126/83"], dtype="object"), "Heart Rate": pd.Series([77], dtype="int64"), "Daily Steps": pd.Series([8000], dtype="int64")})
output_sample = NumpyParameterType([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        # Rename columns
        data = data.rename(columns=lambda x: x.lower().replace(' ', '_'))

        bmi = {
            "Normal": 0,
            "Overweight": 1,
            "Normal Weight": 2,
            "Obese": 3
        }

        sleep_disorder = {
            0: "None",
            1: "Sleep Apnea",
            2: "Insomnia"
        }

        # Clean and one hot encode data
        data.drop("person_id", inplace=True, axis=1)
        data["gender"] = data.gender.apply(lambda s: 1 if s == "Male" else 0)
        occupation = pd.get_dummies(data.occupation, prefix="occupation")
        data.drop("occupation", inplace=True, axis=1)
        data = data.join(occupation)
        data["bmi_category"] = data.bmi_category.map(bmi)
        blood_pressure = pd.get_dummies(data.blood_pressure, prefix="blood_pressure")
        data.drop("blood_pressure", inplace=True, axis=1)
        data = data.join(blood_pressure)

        result = model.predict(data)
        # result.map(sleep_disorder)
        return json.dumps({"result": result})

    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
