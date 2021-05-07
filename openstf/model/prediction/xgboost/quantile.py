# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
import structlog

from openstf.model.prediction.xgboost.xgboost import XGBPredictionModel


class QuantileXGBPredictionModel(XGBPredictionModel):
    def __init__(
        self, pj, forecast_type, trained_model=None, trained_confidence_df=None
    ):
        super().__init__(pj, forecast_type, trained_model, trained_confidence_df)
        self.logger = structlog.get_logger(self.__class__.__name__)

    @property
    def feature_names(self):
        return self.model.feature_names

    def make_forecast(self, forcast_input_data):
        """Make a forecast using a trained XGBoost model.

        Args:
            forcast_input_data (pandas.DataFrame): Forcast (section) input data

        Returns:
            pandas.DataFrame: The forecast DataFrame will have a 'forecast' column and
                a column for eacht percentile in the format:
            'quantile_Pxx' where xx represents the percentile. For example quantile_P50
                is the 50th percentile.

        """
        self.logger.info("Use XGBoost quantile model to make a forecast")

        forecast_request = xgb.DMatrix(forcast_input_data)

        model_forecast = self.model.predict(
            forecast_request, ntree_limit=self.model.best_ntree_limit
        )
        forecast = pd.DataFrame(
            index=forcast_input_data.index, data={"forecast": model_forecast}
        )

        # Loop over quantiles and add quantile forecasts
        for quantile in self.model.quantile_models.keys():
            quantile_key = f"quantile_P{quantile * 100:02.0f}"
            ntree_limit = self.model.quantile_models[quantile].best_ntree_limit
            quantile_forecast = self.model.predict(
                forecast_request, quantile=quantile, ntree_limit=ntree_limit
            )
            forecast[quantile_key] = quantile_forecast
            self.logger.info("Made quantile forecast", quantile=quantile)

        forecast = self._add_standard_deviation_to_forecast(forecast)

        # Add prediction job property to forecast
        forecast = self.add_prediction_job_properties_to_forecast(
            pj=self.pj,
            forecast=forecast,
            algorithm_type=str(self.model_file_path),
            forecast_type=self.forecast_type,
        )

        forecast_length = len(forecast)
        self.logger.info(
            f"Finished making quantile forecast with length '{forecast_length}'",
            forecast_length=forecast_length,
        )

        return forecast

    @staticmethod
    def _add_standard_deviation_to_forecast(forecast):
        """Calculate the mean stdev for a forecast dataframe with different quantiles present.

        Args:
            forecast (pd.DataFrame): DatetimeIndex, columns=['forecast', 'quantile_Pxx', ..., 'quantile_Pnn',]

        Returns:
            pd.DataFrame(DatetimeIndex, columns=['forecast', 'quantile_Pxx', ..., 'quantile_Pnn', 'stdev']

        Raises:
            ValueError: when forecast contains no quantile columns
        """
        standard_deviations = []
        num_quantile_columns = 0

        # calculate standard deviation for each quantile
        for column_name in filter(
            lambda c: c.startswith("quantile_P"), forecast.columns
        ):
            quantile_value = forecast[column_name]
            quantile = float(column_name.split("quantile_P")[1]) / 100.0
            if quantile == 0.5:  # For this value we get inf values
                continue
            stdev = (quantile_value - forecast["forecast"]) / scipy.stats.norm.ppf(
                quantile
            )

            standard_deviations.append(stdev)
            num_quantile_columns += 1

        if num_quantile_columns == 0:
            raise ValueError(
                "Can not calculate standard deviation, forecast does not contain quantile columns"
            )

        # calculate average standard deviation
        standard_deviation = np.nansum(standard_deviations, axis=0) / len(
            standard_deviations
        )
        forecast["stdev"] = standard_deviation

        return forecast

    def predict_fallback(self, forecast_index, load):
        """Overwrite default fallbackforecast to include quantiles.
        Use default fallbackforecast as startingpoint
        Quantiles are based on percentiles of hourly-grouped load_data
        Which quantiles are calculated are hard-coded for now
        TODO: make hardcoded quantiles not hardcoded

        Args:
            - forecast_index: pd.DatetimeIndex, used to determine new forecast timerange
            - load: pd.DataFrame(DatetimeIndex, columns=[load])

        Returns:
            pd.DataFrame(DatetimeIndex, forecast, quantile_Px,...)
        """

        # Use default
        forecast = super().predict_fallback(forecast_index, load)

        # Use the forecasted load since the actual load contains nan timestamps
        # in the future
        load = forecast[["forecast"]].rename(columns={"forecast": "load"})

        # Calculate quantiles per hour of day - list with items <1 can be copied from database
        quantiles = [x * 100 for x in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]]
        q_vals = load.groupby(load.index.hour).apply(
            lambda x: np.percentile(x, quantiles)
        )

        q_vals = pd.DataFrame(
            q_vals.values.tolist(),
            index=q_vals.index,
            columns=[f"quantile_P{quantile:02.0f}" for quantile in quantiles],
        )

        # Add quantiles to the forecast
        forecast = forecast.merge(q_vals, left_on=forecast.index.hour, right_index=True)
        forecast = forecast.loc[:, [x for x in forecast.columns if x != "key_0"]]
        return forecast
