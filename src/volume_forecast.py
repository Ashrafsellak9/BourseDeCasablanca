"""
Prévision du volume marché (journalier) pour contextualiser les observations.

- **ARIMA** : ``statsmodels`` (plusieurs ordres testés, choix par AIC).
- **Prophet** : optionnel si le paquet ``prophet`` est installé (saisonnalité hebdomadaire, fréquence séances).

Les volumes sont modélisés en **M MAD** (÷ 1e6) pour la stabilité numérique, puis reconvertis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SCALE = 1e6
MIN_POINTS_ARIMA = 18
MIN_POINTS_PROPHET = 20


@dataclass
class VolumeForecastResult:
    """Sortie standardisée pour le graphique et les légendes."""

    history_dates: pd.DatetimeIndex
    history_actual: np.ndarray
    history_fitted: np.ndarray | None
    forecast_dates: pd.DatetimeIndex
    forecast_mean: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    method: str
    message: str
    success: bool


def _business_days_after(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last + pd.tseries.offsets.BDay(1), periods=n)


def _naive_forecast(
    y: pd.Series,
    horizon: int,
    message: str,
) -> VolumeForecastResult:
    last = y.index.max()
    idx = _business_days_after(last, horizon)
    last_v = float(y.iloc[-1])
    mu = float(y.tail(min(10, len(y))).mean())
    mean = np.full(horizon, mu, dtype=float)
    std = float(y.tail(min(20, len(y))).std()) or abs(mu) * 0.05 or 1.0
    lo = np.maximum(mean - 1.28 * std, 0)
    hi = mean + 1.28 * std
    return VolumeForecastResult(
        history_dates=y.index,
        history_actual=y.values.astype(float),
        history_fitted=None,
        forecast_dates=idx,
        forecast_mean=mean,
        forecast_lower=lo,
        forecast_upper=hi,
        method="Naïf (moyenne récente)",
        message=message,
        success=True,
    )


def _fit_arima(y_m: pd.Series, horizon: int) -> VolumeForecastResult | None:
    from statsmodels.tsa.arima.model import ARIMA

    orders = [(1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 0, 1), (0, 1, 1), (2, 1, 0)]
    best_aic = np.inf
    best_res = None
    for order in orders:
        try:
            m = ARIMA(y_m, order=order)
            res = m.fit(method_kwargs={"warn_convergence": False})
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_res = res
        except Exception:
            continue
    if best_res is None:
        return None

    try:
        fitted = best_res.predict(start=0, end=len(y_m) - 1)
        fc = best_res.get_forecast(steps=horizon)
        mean_m = fc.predicted_mean.values.astype(float)
        ci = fc.conf_int(alpha=0.2)
        lo_m = ci.iloc[:, 0].values.astype(float)
        hi_m = ci.iloc[:, 1].values.astype(float)
    except Exception:
        return None

    last = y_m.index.max()
    f_idx = _business_days_after(last, horizon)
    mo = getattr(best_res, "model_orders", {}) or {}
    msg = f"ARIMA {mo} — AIC={best_aic:.0f} (volume en M MAD)."
    return VolumeForecastResult(
        history_dates=y_m.index,
        history_actual=(y_m * SCALE).values,
        history_fitted=(np.asarray(fitted, dtype=float) * SCALE),
        forecast_dates=f_idx,
        forecast_mean=mean_m * SCALE,
        forecast_lower=np.maximum(lo_m * SCALE, 0),
        forecast_upper=hi_m * SCALE,
        method="ARIMA",
        message=msg,
        success=True,
    )


def _fit_prophet(y: pd.Series, horizon: int) -> VolumeForecastResult | None:
    try:
        from prophet import Prophet
    except ImportError:
        return None

    if len(y) < MIN_POINTS_PROPHET:
        return None

    df_p = pd.DataFrame({"ds": pd.to_datetime(y.index).normalize(), "y": y.values.astype(float) / SCALE})
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.08,
    )
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon, freq="B", include_history=True)
    pred = m.predict(future)
    pred["ds"] = pd.to_datetime(pred["ds"]).dt.normalize()
    last_ds = df_p["ds"].max()
    fc_rows = pred[pred["ds"] > last_ds].drop_duplicates("ds").sort_values("ds").head(horizon)
    if len(fc_rows) < horizon:
        return None

    hist_df = pd.DataFrame({"ds": df_p["ds"], "actual": df_p["y"].values})
    merged = hist_df.merge(
        pred[["ds", "yhat"]].drop_duplicates("ds"),
        on="ds",
        how="left",
    )
    fitted_m = merged["yhat"].ffill().bfill().values.astype(float)

    return VolumeForecastResult(
        history_dates=pd.DatetimeIndex(df_p["ds"]),
        history_actual=(df_p["y"].values.astype(float) * SCALE),
        history_fitted=fitted_m * SCALE,
        forecast_dates=pd.DatetimeIndex(fc_rows["ds"].values),
        forecast_mean=(fc_rows["yhat"].values.astype(float) * SCALE),
        forecast_lower=np.maximum(fc_rows["yhat_lower"].values.astype(float) * SCALE, 0),
        forecast_upper=fc_rows["yhat_upper"].values.astype(float) * SCALE,
        method="Prophet",
        message="Prophet (tendance + saisonnalité hebdomadaire, volume en M MAD).",
        success=True,
    )


def fit_volume_forecast(
    volume_series: pd.Series,
    *,
    method: Literal["arima", "prophet"] = "arima",
    horizon: int = 7,
) -> VolumeForecastResult:
    """
    Ajuste un modèle sur ``volume_series`` (index = dates, valeurs = Volume MAD).

    Parameters
    ----------
    method
        ``arima`` ou ``prophet`` (Prophet renvoie naïf si paquet absent ou échec).
    horizon
        Nombre de **jours ouvrés** à prévoir après la dernière observation.
    """
    y = volume_series.dropna().sort_index().astype(float)
    y = y[~y.index.duplicated(keep="last")]
    if len(y) < 5:
        return VolumeForecastResult(
            history_dates=y.index,
            history_actual=y.values if len(y) else np.array([]),
            history_fitted=None,
            forecast_dates=pd.DatetimeIndex([]),
            forecast_mean=np.array([]),
            forecast_lower=np.array([]),
            forecast_upper=np.array([]),
            method="—",
            message="Pas assez de points pour une prévision.",
            success=False,
        )

    horizon = int(max(1, min(horizon, 30)))

    if method == "prophet":
        r = _fit_prophet(y, horizon)
        if r is not None:
            return r
        return _naive_forecast(
            y,
            horizon,
            "Prophet indisponible ou non installé ; moyenne mobile utilisée. "
            "Installez le paquet `prophet` pour l'activer.",
        )

    # ARIMA
    if len(y) < MIN_POINTS_ARIMA:
        return _naive_forecast(
            y,
            horizon,
            f"Série courte (n={len(y)}) ; prévision naïve (moyenne des {min(10, len(y))} derniers jours).",
        )

    y_m = y / SCALE
    r = _fit_arima(y_m, horizon)
    if r is not None:
        return r
    return _naive_forecast(y, horizon, "ARIMA non convergent ; prévision naïve utilisée.")
