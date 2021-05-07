from datetime import datetime, timedelta
from pathlib import Path
import structlog
import joblib
from ktpbase.database import DataBase
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.confidence_interval_generator import ConfidenceIntervalGenerator
from openstf.model.model_creator import ModelCreator
from openstf.model.reporter import Reporter
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.validation.validation import validate, clean, is_data_sufficient

TRAIN_HORIZONS: list[float] = [0.25, 24.0]
MAXIMUM_MODEL_AGE: float = 7

EARLY_STOPPING_ROUNDS: int = 10
PENALTY_FACTOR_OLD_MODEL: float = 1.2

SAVE_PATH = Path('.')
OLD_MODEL_PATH = '.'


def train_model_pipeline(pj: dict, check_old_model_age: bool = True,
                         compare_to_old: bool = True) -> None:

    logger = structlog.get_logger(__name__)
    logger.info(
        "Start training model",
        prediction_id=pj["id"],
        customer_name=pj["name"]
)

    # Initialize database
    db = DataBase()

    # Get old model path and age
    # TODO some function here that retrieves age of the old model
    old_model_age = 5

    # Check old model age and continue yes/no
    if (old_model_age > MAXIMUM_MODEL_AGE) and check_old_model_age:
        return

    # Get hyper parameters
    hyper_params = db.get_hyper_params(pj)

    # Get input data
    input_data = db.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime.utcnow() - timedelta(
            days=90
        ),
        datetime_end=datetime.utcnow(),
    )

    # Validate and clean data
    validated_data = clean(validate(input_data))

    # Check if sufficient data is left after cleaning
    if not is_data_sufficient(validated_data):
        raise (RuntimeError(
            f"Not enough data left after validation for {pj['name']}, check input data!"))

    # Add features
    data_with_features = TrainFeatureApplicator(TRAIN_HORIZONS,
                                                features=db.get_featureset(
                                                    hyper_params["featureset_name"])
                                                ).add_features(validated_data)

    # Split data
    train_data, validation_data, test_data = split_data_train_validation_test(
        data_with_features.sort_index(axis=1))

    # Create relevant model
    model = ModelCreator(pj).create_model()

    # Configure evals for early stopping
    eval_set = [(train_data.iloc[:, 1:], train_data.iloc[:, 0]),
                (validation_data.iloc[:, 1:], validation_data.iloc[:, 0])]

    model.set_params(params=hyper_params)
    model.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0],
              eval_set=eval_set,
              early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    # Check if new model is better than old model
    if compare_to_old:
        old_model = joblib.load(OLD_MODEL_PATH)
        combined = train_data.append(validation_data)

        # Score method always returns R^2
        score_new_model = model.score(combined.iloc[:, 1:], combined.iloc[:, 0])
        score_old_model = old_model.score(combined.iloc[:, 1:], combined.iloc[:, 0])

        # Check if R^2 is better for old model
        if score_old_model > score_new_model * PENALTY_FACTOR_OLD_MODEL:
            logger.warning(f"Old model is better than new model for {pj['name']}")
        else:
            logger.info(
                "New model is better than old model, continuing with training procces")

    # Report about the training procces
    Reporter(pj, train_data, validation_data, test_data)\
        .make_and_save_dashboard_figures(model, SAVE_PATH)

    # Do confidence interval determination
    model = ConfidenceIntervalGenerator(pj, validation_data)\
        .generate_confidence_interval_data(model)

    # Persist model
    joblib.dump(model, SAVE_PATH / "model.sav")

if __name__ == "__main__":
    pj = DataBase().get_prediction_job(307)

    train_model_pipeline(pj, False, False)
