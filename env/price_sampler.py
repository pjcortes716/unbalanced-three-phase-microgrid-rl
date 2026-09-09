"""Perfil económico exógeno y forecasts de precio para la microrred.

El perfil diario de este módulo es una hipótesis importada deliberadamente de
otro trabajo previo; no procede del paper de la microrred industrial. No se
reescalan sus valores, no se añade el recargo de 0.12 EUR/kWh del paper y no
se distinguen todavía precios de compra y venta.
"""

from __future__ import annotations

import operator

import numpy as np


PRICE_PROFILE_24H = np.array([
    1.00,
    0.99,
    0.95,
    0.95,
    0.96,
    1.00,
    5.48,
    9.95,
    10.13,
    10.06,
    9.88,
    7.46,
    5.04,
    5.00,
    4.96,
    7.46,
    9.95,
    10.13,
    10.06,
    9.88,
    5.49,
    1.10,
    1.06,
    1.00,
], dtype=float)
PRICE_SCALE = float(np.max(PRICE_PROFILE_24H))


class PriceSampler:
    """Muestrea precios reales y forecasts a partir de un perfil base diario.

    ``profile_24h`` representa la media determinista, no una realización. En
    cada ``reset`` se genera una única semana de precios reales con
    ``sigma = 0.10 * base_price``. A continuación se generan forecasts
    independientes alrededor de esas realizaciones con
    ``sigma = 0.10 * abs(real_price)``. Las consultas son idempotentes y no
    existe un reloj interno que avance automáticamente.
    """

    def __init__(
        self,
        profile_24h: np.ndarray = PRICE_PROFILE_24H,
        *,
        horizon_hours: int = 168,
        forecast_h: int = 11,
        realization_sigma_frac: float = 0.10,
        forecast_sigma_frac: float = 0.10,
        seed: int | None = 0,
    ) -> None:
        profile = np.asarray(profile_24h, dtype=float)
        if profile.shape != (24,) or not np.all(np.isfinite(profile)) or np.any(profile <= 0.0):
            raise ValueError("profile_24h debe tener 24 precios positivos y finitos.")
        if horizon_hours <= 0 or forecast_h <= 0:
            raise ValueError("horizon_hours y forecast_h deben ser positivos.")
        if realization_sigma_frac < 0.0 or not np.isfinite(realization_sigma_frac):
            raise ValueError("realization_sigma_frac debe ser finito y no negativo.")
        if forecast_sigma_frac < 0.0 or not np.isfinite(forecast_sigma_frac):
            raise ValueError("forecast_sigma_frac debe ser finito y no negativo.")

        self.profile_24h = profile.copy()
        self.horizon_hours = int(horizon_hours)
        self.forecast_h = int(forecast_h)
        self.realization_sigma_frac = float(realization_sigma_frac)
        self.forecast_sigma_frac = float(forecast_sigma_frac)
        self.rng = np.random.default_rng(seed)
        self.base_price: np.ndarray | None = None
        self.base_price_ext: np.ndarray | None = None
        self.price_real: np.ndarray | None = None
        self.price_real_ext: np.ndarray | None = None
        self.price_forecast_ext: np.ndarray | None = None
        self.realization_noise: np.ndarray | None = None
        self.forecast_noise_ext: np.ndarray | None = None

    def _require_prepared_profile(self) -> None:
        if self.price_real_ext is None or self.price_forecast_ext is None:
            raise RuntimeError("Llama reset() antes de consultar el perfil de precio.")

    def _validate_time_index(self, t: int) -> int:
        try:
            t = operator.index(t)
        except TypeError as exc:
            raise TypeError("t debe ser un índice entero de hora.") from exc
        if not 0 <= t < self.horizon_hours:
            raise IndexError(f"t={t} está fuera del horizonte [0, {self.horizon_hours - 1}].")
        return int(t)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Prepara las 168 realizaciones y los forecasts de borde.

        Si se facilita ``seed``, reinicia únicamente el RNG local del precio.
        Se muestrean una vez los precios reales de la semana y luego el error
        de forecast. La extensión de once horas reutiliza cíclicamente las
        primeras realizaciones semanales para conservar ``forecast_at`` en el
        borde del episodio sin generar ruido durante las consultas.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        repetitions = int(np.ceil(self.horizon_hours / self.profile_24h.size))
        self.base_price = np.tile(self.profile_24h, repetitions)[: self.horizon_hours]
        realization_sigma = self.realization_sigma_frac * self.base_price
        self.realization_noise = self.rng.normal(0.0, realization_sigma).astype(float)
        self.price_real = self.base_price + self.realization_noise

        self.base_price_ext = np.concatenate((self.base_price, self.base_price[: self.forecast_h]))
        self.price_real_ext = np.concatenate((self.price_real, self.price_real[: self.forecast_h]))
        forecast_sigma = self.forecast_sigma_frac * np.abs(self.price_real_ext)
        self.forecast_noise_ext = self.rng.normal(0.0, forecast_sigma).astype(float)
        # No se recorta: el Box de observación permite forecasts finitos fuera
        # del máximo nominal, igual que los demás forecasts exógenos.
        self.price_forecast_ext = self.price_real_ext + self.forecast_noise_ext
        return (
            self.price_real.copy(),
            self.price_forecast_ext.copy(),
        )

    def value_at(self, t: int) -> float:
        """Devuelve el precio real muestreado de ``t`` sin mutar estado."""
        self._require_prepared_profile()
        t = self._validate_time_index(t)
        return float(self.price_real_ext[t])

    def forecast_at(self, t: int, horizon: int = 11) -> np.ndarray:
        """Devuelve el forecast futuro ``[t+1, ..., t+horizon]`` sin mutar."""
        self._require_prepared_profile()
        t = self._validate_time_index(t)
        try:
            horizon = operator.index(horizon)
        except TypeError as exc:
            raise TypeError("horizon debe ser un entero positivo.") from exc
        if horizon <= 0:
            raise ValueError("horizon debe ser mayor que cero.")
        last_index = t + horizon
        if last_index >= len(self.price_forecast_ext):
            raise IndexError(
                f"Forecast [t+1, ..., t+{horizon}] no disponible para t={t}; "
                f"el último índice preparado es {len(self.price_forecast_ext) - 1}."
            )
        return self.price_forecast_ext[t + 1 : last_index + 1].astype(float, copy=True)
