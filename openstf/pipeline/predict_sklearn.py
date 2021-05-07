from datetime import datetime, timedelta
from pathlib import Path
import structlog
import joblib
import pandas as pd


from ktpbase.database import DataBase
from openstf.pipeline.create_forecast import generate_forecast_datetime_range, generate_inputdata_datetime_range
from openstf.validation.validation import validate, clean, is_data_sufficient
from openstf.feature_engineering.feature_applicator import OperationalPredictFeatureApplicator
from openstf.model.confidence_interval_applicator import ConfidenceIntervalApplicator
MODEL_LOCATION = Path('.')

def predict_pipeline(pj):

    logger = structlog.get_logger(__name__)
    logger.info(
        "Start making prediction",
        prediction_id=pj["id"],
        customer_name=pj["name"],
        prediction_type="prediction",
    )

    # Get input data
    forecast_start, forecast_end = generate_forecast_datetime_range(
        resolution_minutes=pj["resolution_minutes"],
        horizon_minutes=pj["horizon_minutes"],
    )
    datetime_start, datetime_end = generate_inputdata_datetime_range(
        t_behind_days=14, t_ahead_days=3
    )

    input_data = DataBase().get_model_input(
        pid=pj["id"],
        location=(pj["lat"], pj["lon"]),
        # TODO it makes more sense to do the string converion in the
        # DataBase class and use pure Python datatypes in the rest of the code
        datetime_start=str(datetime_start),
        datetime_end=str(datetime_end),
    )

    # Get model
    model = joblib.load(MODEL_LOCATION / "model.sav")

    # Get hyper parameters
    hyper_params = DataBase().get_hyper_params(pj)

    # Validate and clean data
    validated_data = validate(input_data)

    # Add features
    data_with_features = OperationalPredictFeatureApplicator(horizons=[0.25], features=model._Booster.feature_names).add_features(validated_data)

    # Check if sufficient data is left after cleaning
    if not is_data_sufficient(data_with_features):
        logger.warning('Use fallback model')

    # Predict
    forecast_input_data = data_with_features[forecast_start:forecast_end]

    model_forecast = model.predict(forecast_input_data.sort_index(axis=1))

    forecast = pd.DataFrame(
        index=forecast_input_data.index, data={"forecast": model_forecast}
    )

    # Add confidence
    forecast = ConfidenceIntervalApplicator(model).add_confidence_interval(forecast)

    # Prepare for output
    forecast = add_prediction_job_properties_to_forecast(pj, forecast,)

    # write forefast to db
    print(forecast)

def add_prediction_job_properties_to_forecast(
    pj, forecast, forecast_type=None, forecast_quality=None
):
    # self.logger.info("Postproces in preparation of storing")
    if forecast_type is None:
        forecast_type = pj["typ"]
    else:
        # get the value from the enum
        forecast_type = forecast_type.value

    # NOTE this field is only used when making the babasecase forecast and fallback
    if forecast_quality is not None:
        forecast["quality"] = forecast_quality

    # TODO rename prediction job typ to type
    # TODO algtype = model_file_path, perhaps we can find a more logical name
    # TODO perhaps better to make a forecast its own class!
    # TODO double check and sync this with make_basecase_forecast (other fields are added)
    # !!!!! TODO fix the requirement for customer
    forecast["pid"] = pj["id"]
    forecast["customer"] = pj["name"]
    forecast["description"] = pj["description"]
    forecast["type"] = forecast_type
    forecast["algtype"] = pj['model']

    return forecast


if __name__ == "__main__":
    pj = DataBase().get_prediction_job(307)
    predict_pipeline(pj)