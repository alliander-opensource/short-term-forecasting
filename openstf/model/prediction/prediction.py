# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from openstf.enums import ForecastType, MLModelType
from openstf.model.serializer.creator import ModelSerializerCreator

FLATLINER_THRESHOLD = 6


class AbstractPredictionModel(ABC):
    def __init__(
        self, pj, forecast_type, trained_model=None, trained_confidence_df=None
    ):
        super().__init__()
        self.pj = pj
        self.model_type = MLModelType(pj["model"])
        self.forecast_type = forecast_type
        self.serializer = ModelSerializerCreator.create_model_serializer(
            model_type=self.model_type
        )
        self.confidence_df = trained_confidence_df
        self.model = trained_model
        self.model_file_path = None

        if self._bool_load_model_from_disk():
            self.model, self.model_file_path = self.serializer.load(pid=self.pj["id"])

    def _bool_load_model_from_disk(self):
        """Determine if model should be loaded from disk"""
        return self.forecast_type is not ForecastType.BASECASE and self.model is None

    @property
    @abstractmethod
    def feature_names(self):
        pass

    @abstractmethod
    def calculate_completeness(self, forcast_input_data):
        pass

    @abstractmethod
    def make_forecast(self, forcast_input_data):
        pass

    def make_fallback_forecast(self, forecast_input_data, load_data):
        if self.forecast_type is ForecastType.BASECASE:
            raise ValueError(
                "Can't make a fallback prediction for forecast type basecase"
            )

        self.logger.warning(
            "Using fallback forecast (*high forecast*)",
            forecast_type="fallback",
            pid=self.pj["id"],
        )
        forecast = self.predict_fallback(
            forecast_index=forecast_input_data.index, load=load_data
        )

        # Add corrections (if file exists and is not empty)
        forecast = self._add_standard_deviation_to_forecast(forecast)
        # Add prediciont job propertie to forecast
        forecast = self.add_prediction_job_properties_to_forecast(
            pj=self.pj,
            forecast=forecast,
            algorithm_type=str(self.model_file_path),
            forecast_type=self.forecast_type,
        )
        return forecast

    @staticmethod
    def make_basecase_forecast(pj, historic_load, overwrite_delay_hours=48):
        """Make a 'basecase' forecast

            Result is writen to database for all forecasts further in time than
            overwrite_delay_hours. Idea is that if all else fails, this forecasts is
            still available.

            Basecase example: the load of last week.

            As 'quality', the value 'not_renewed' is used.

        Args:
            historic_load (pd.DataFrame): Historic load
            overwrite_delay_hours (float): times before this in the future are not
                forecasted

        Returns:
            pd.DataFrame: Basecase forecast (which was written to the database)
        """
        # - Make forecast
        # Make basecase forecast: Use load of last week
        basecase_forecast = historic_load.shift(7, "D")

        # Maybe there is still missing data, for example if the cdb has been down for a
        # while in this case, use the load of 2 weeks before
        basecase_forecast = basecase_forecast.append(historic_load.shift(14, "D"))
        basecase_forecast = basecase_forecast[
            np.invert(basecase_forecast.index.duplicated())
        ]

        # - Post Process
        # Don't update first 48 hours
        forecast_start = datetime.now(timezone.utc) + timedelta(
            hours=overwrite_delay_hours
        )
        basecase_forecast = basecase_forecast[forecast_start:]
        # Don't update nan values
        basecase_forecast.dropna(inplace=True)
        # rename
        basecase_forecast = basecase_forecast.rename(columns=dict(load="forecast"))

        # Also make a basecase forecast for the forecast_other component. This will make a
        # simple basecase components forecast available and ensures that the sum of
        # the components (other, wind and solar) is equal to the normal basecase forecast
        basecase_forecast["forecast_other"] = basecase_forecast["forecast"]

        # Estimate the stdev a bit smart
        # use the stdev of the hour for historic_load
        std_per_hour = (
            historic_load.groupby(historic_load.index.hour)
            .std()
            .rename(columns=dict(load="stdev"))
        )
        basecase_forecast["hour"] = basecase_forecast.index.hour
        basecase_forecast = basecase_forecast.merge(
            std_per_hour, left_on="hour", right_index=True
        )
        del basecase_forecast["hour"]

        # - Add properties to forecast
        basecase_forecast = (
            AbstractPredictionModel.add_prediction_job_properties_to_forecast(
                pj=pj,
                forecast=basecase_forecast,
                algorithm_type="basecase_lastweek",
                forecast_quality="not_renewed",
            )
        )

        return basecase_forecast.sort_index()

    @staticmethod
    def predict_fallback(forecast_index, load):
        """Make a fall back forecast
            Use historic profile of most extreme day.

            Set the value of the forecast 'quality' column to 'substituted'

        Args:
            forecast_index (pandas.DatetimeIndex): Index required for the forecast
            load (pandas.DataFrame): load

        Raises:
            RuntimeError: When the most important feature does not start with
                T-, wind or radi

        Returns:
            pandas.DataFrame: Fallback forecast DataFrame with columns:
                'forecast', 'quality'
        """
        # Check if load is completely empty
        if len(load.dropna()) == 0:
            raise ValueError("No historic load data available")

        # Find most extreme historic day (do not count today as it is incomplete)
        day_with_highest_load_date = (
            load[load.index.tz_localize(None).date != datetime.utcnow().date()]
            .idxmax()
            .load.date()
        )
        # generate datetime range of the day with the highest load
        from_datetime = pd.Timestamp(day_with_highest_load_date, tz=load.index.tz)
        till_datetime = from_datetime + pd.Timedelta("1 days")

        # slice load dataframe, all rows for the day with the highest load
        highest_daily_loadprofile = load.loc[
            (load.index >= from_datetime) & (load.index < till_datetime)
        ]

        highest_daily_loadprofile["time"] = highest_daily_loadprofile.index.time

        forecast = pd.DataFrame(index=forecast_index)
        forecast["time"] = forecast.index.time
        forecast = (
            forecast.reset_index()
            .merge(
                highest_daily_loadprofile, left_on="time", right_on="time", how="outer"
            )
            .set_index("index")
        )
        forecast = forecast[["load"]].rename(columns=dict(load="forecast"))

        # Add a column quality.
        forecast["quality"] = "substituted"

        return forecast

    def _get_standard_deviation(self):

        # 1. used supplied (constructor) standard deviation if not None
        if self.confidence_df is not None:
            self.logger.info("Using supplied confidence_df as standard deviation")
            return self.confidence_df

        # 2. load standard deviation from disk (stored after training model)
        self.logger.info("Loading corrections from disk")
        corrections_file_path = (
            self.model_file_path.parent / f"{self.serializer.CORRECTIONS_FILENAME}"
        )
        try:
            corrections = pd.read_csv(corrections_file_path, index_col=0)
        except FileNotFoundError as e:
            self.logger.warning(
                "Did not add corrections, file not found. Try retraining the model "
                "with cor = True",
                exc_info=e,
            )
            return None

        if len(corrections) != len(corrections.dropna()):
            self.logger.warning("Correction file contains no values!")
            return None

        return corrections

    def _add_standard_deviation_to_forecast(self, forecast):
        """Add a standard deviation to a live forecast.

        The stdev for intermediate forecast horizons is interpolated.

        For the input standard_deviation, it is preferred that the forecast horizon is
        expressed in Hours, instead of 'Near/Far'. For now, Near/Far is supported,
        but this will be deprecated.

        Args:
            forecast (pd.DataFrame): Forecast DataFram with columns: "forecast"
            standard_deviation (pd.DataFrame): Standard deviation. DataFrame with columns:
                "hour", "horizon", "stdev"
            (optional) interpolate (str): Interpolation method, options: "exponential" or "linear"

        Returns:
            pd.DataFrame: Forecast with added standard deviation. DataFrame with columns:
                "forecast", "stdev"
        """

        if self.forecast_type is ForecastType.BASECASE:
            raise ValueError(
                "Can not make forecast corrections for forecast type basecase"
            )
        standard_deviation = self._get_standard_deviation()

        if standard_deviation is None:
            return forecast

        # -------- Moved from feature_engineering.add_stdev ------------------------- #
        # pivot
        stdev = standard_deviation.pivot_table(columns=["horizon"], index="hour")[
            "stdev"
        ]
        # Prepare input dataframes
        # Rename Near and Far to 0.25 and 47 respectively, if present.
        # Timehorizon in hours is preferred to less descriptive Near/Far
        if "Near" in stdev.columns:
            near = (forecast.index[1] - forecast.index[0]).total_seconds() / 3600.0
            # Try to infer for forecast df, else use a max of 48 hours
            far = min(
                48.0,
                (forecast.index.max() - forecast.index.min()).total_seconds() / 3600.0,
            )
            stdev.rename(columns={"Near": near, "Far": far}, inplace=True)
        else:
            near = stdev.columns.min()
            far = stdev.columns.max()

        forecast_copy = forecast.copy(deep=True)
        # add time ahead column if not already present
        if "tAhead" not in forecast_copy.columns:
            # Assume first datapoint is 'now'
            forecast_copy["tAhead"] = (
                forecast_copy.index - forecast_copy.index.min()
            ).total_seconds() / 3600.0
        # add helper column hour
        forecast_copy["hour"] = forecast_copy.index.hour

        # Define functions which can be used to approximate the error for in-between time horizons
        # Let's fit and exponential decay of accuracy
        def calc_exp_dec(t, stdev_row, near, far):
            # We use the formula sigma(t) = (1 - A * exp(-t/tau)) + b
            # Strictly speaking, tau is specific for each time series.
            # However, for simplicity, we use tau = Far/4.
            # This represents a situation where the stdev at 25% of the Far horizon,
            # has increased by two.
            tau = far / 4.0
            # Filling in the known sigma(Near) and sigma(Far) gives:
            sf, sn = stdev_row[far], stdev_row[near]
            A = (sf - sn) / ((1 - np.exp(-far / tau)) - (1 - np.exp(-near / tau)))
            b = sn - A * (1 - np.exp(-near / tau))
            return A * (1 - np.exp(-t / tau)) + b

        # Add stdev to forecast_copy dataframe
        forecast_copy["stdev"] = forecast_copy.apply(
            lambda x: calc_exp_dec(x.tAhead, stdev.loc[x.hour], near, far), axis=1
        )
        # -------- End moved from feature_engineering.add_stdev --------------------- #

        self.logger.info("Corrections added to forecast")

        return forecast_copy.drop(columns=["hour"])

    # NOTE idea is that this can perhaps be removed since its not required to make a
    # prediction but more specific to how we (Alliander/KTP) store things
    @staticmethod
    def add_prediction_job_properties_to_forecast(
        pj, forecast, algorithm_type, forecast_type=None, forecast_quality=None
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
        forecast["algtype"] = algorithm_type

        return forecast
