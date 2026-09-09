import operator
from pathlib import Path

import numpy as np

class OfficeLoadSampler:
    """
    - Carga pool de semanas (n_weeks, 168)
    - La semana seleccionada y escalada es el perfil base histórico.
    - En ``reset`` se genera una única realización física desde esa base y
      luego un forecast independiente alrededor de la realización.
    - Pmax es una escala histórica, no un techo de la demanda estocástica.
    """

    def __init__(
        self,
        filepath: str | Path = "office_weeks_pool.npz",
        seed: int = 0,
        pmax_target_kw: float = 18.74,
        horizon_hours: int = 168,
        forecast_h: int = 11,
        realization_sigma_frac: float = 0.10,
        forecast_sigma_frac: float = 0.10,
        week_noise_sigma: float = 0.03,
        week_noise_phi: float = 0.85,
        clip_nonnegative: bool = True,
    ):
        self.filepath = filepath

        
        self.horizon_hours = int(horizon_hours)
        self.forecast_h = int(forecast_h)

        self.n_weeks_original = 0
        self.n_weeks_discarded = 0
        self.discarded_week_indices = np.array([], dtype=int)
        self.valid_week_indices = np.array([], dtype=int)
        self.weeks_pool = self._load_weeks_pool(filepath)

        self.rng = np.random.default_rng(seed)
        self.pmax_target_kw = float(pmax_target_kw)

        self.realization_sigma_frac = float(realization_sigma_frac)
        self.forecast_sigma_frac = float(forecast_sigma_frac)
        self.week_noise_sigma = float(week_noise_sigma)
        self.week_noise_phi = float(week_noise_phi)
        self.clip_nonnegative = bool(clip_nonnegative)
        if self.realization_sigma_frac < 0.0 or not np.isfinite(self.realization_sigma_frac):
            raise ValueError("realization_sigma_frac debe ser finito y no negativo.")
        if self.forecast_sigma_frac < 0.0 or not np.isfinite(self.forecast_sigma_frac):
            raise ValueError("forecast_sigma_frac debe ser finito y no negativo.")

        self.t = 0
        self.week_idx = None
        self.original_week_idx = None
        self.scale = None

        self.P_base_profile = None
        self.P_base_ext = None
        self.P_real_profile = None
        self.P_real_ext = None
        self.P_forecast_ext = None
        self.realization_noise = None
        self.forecast_noise_ext = None
        self._week_rng = None
        self._realization_rng = None
        self._forecast_rng = None


    def _load_weeks_pool(self, filepath: str | Path) -> np.ndarray:
        with np.load(filepath, allow_pickle=True) as data:
            weeks = data["weeks"].astype(float)
        assert weeks.ndim == 2 and weeks.shape[1] == self.horizon_hours, \
            f"weeks debe ser (n_weeks, {self.horizon_hours})"

        self.n_weeks_original = int(weeks.shape[0])
        valid_rows = np.isfinite(weeks).all(axis=1)
        self.discarded_week_indices = np.flatnonzero(~valid_rows)
        self.valid_week_indices = np.flatnonzero(valid_rows)
        self.n_weeks_discarded = int(self.discarded_week_indices.size)

        weeks = weeks[valid_rows]
        if weeks.shape[0] == 0:
            raise ValueError(
                f"El pool '{filepath}' no contiene semanas finitas utilizables "
                f"para un horizonte de {self.horizon_hours} horas."
            )
        return weeks

    def _make_forecast(self, P_real_ext: np.ndarray) -> np.ndarray:
        """
        Forecast pointwise:
          P_fc = P_real + N(0, sigma),
        donde sigma = forecast_sigma_frac * abs(P_real).

        El forecast es una predicción y no se recorta, aunque la realización
        física sí se protege contra valores negativos.
        """
        sigma = self.forecast_sigma_frac * np.abs(P_real_ext)
        noise = self._forecast_rng.normal(0.0, sigma)
        P_fc = P_real_ext + noise
        self.forecast_noise_ext = noise.astype(float, copy=True)
        return P_fc

    def _configure_rng(self, *, rng=None, seed: int | None = None) -> None:
        """Actualiza el RNG solo cuando el llamador lo solicita explícitamente."""
        if rng is not None and seed is not None:
            raise ValueError("Pasa rng o seed, no ambos.")
        if rng is not None:
            if not all(hasattr(rng, method) for method in ("integers", "normal")):
                raise TypeError("rng debe ofrecer los métodos integers() y normal() de NumPy.")
            self.rng = rng
            entropy = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif seed is not None:
            try:
                entropy = operator.index(seed)
            except TypeError as exc:
                raise TypeError("seed debe ser un entero o None.") from exc
            self.rng = np.random.default_rng(entropy)
        else:
            entropy = int(self.rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        week_seq, realization_seq, forecast_seq = np.random.SeedSequence(entropy).spawn(3)
        self._week_rng = np.random.default_rng(week_seq)
        self._realization_rng = np.random.default_rng(realization_seq)
        self._forecast_rng = np.random.default_rng(forecast_seq)

    def _require_prepared_profiles(self) -> None:
        if self.P_real_ext is None or self.P_forecast_ext is None:
            raise RuntimeError("Debes llamar reset() antes de consultar perfiles de carga.")

    def _validate_time_index(self, t: int) -> int:
        try:
            t = operator.index(t)
        except TypeError as exc:
            raise TypeError("t debe ser un índice entero de hora.") from exc
        if not 0 <= t < self.horizon_hours:
            raise IndexError(f"t={t} está fuera del horizonte [0, {self.horizon_hours - 1}].")
        return int(t)

    def value_at(self, t: int) -> float:
        """Devuelve la demanda real de la hora ``t`` sin mutar ningún estado."""
        self._require_prepared_profiles()
        t = self._validate_time_index(t)
        return float(self.P_real_ext[t])

    def forecast_at(self, t: int, horizon: int | None = None) -> np.ndarray:
        """Devuelve el forecast futuro ``[t+1, ..., t+horizon]`` sin avanzar tiempo.

        La extensión cíclica creada en :meth:`reset` permite conservar una
        longitud fija cerca del final de la semana; es una aproximación de
        borde del episodio, no una nueva semana observada.
        """
        self._require_prepared_profiles()
        t = self._validate_time_index(t)
        if horizon is None:
            horizon = self.forecast_h
        try:
            horizon = operator.index(horizon)
        except TypeError as exc:
            raise TypeError("horizon debe ser un entero positivo.") from exc
        if horizon <= 0:
            raise ValueError("horizon debe ser mayor que cero.")

        last_index = t + horizon
        if last_index >= len(self.P_forecast_ext):
            raise IndexError(
                f"Forecast [t+1, ..., t+{horizon}] no disponible para t={t}; "
                f"el último índice preparado es {len(self.P_forecast_ext) - 1}."
            )
        return self.P_forecast_ext[t + 1 : last_index + 1].astype(float, copy=True)

    def reset(self, *, rng=None, seed: int | None = None, week_idx: int | None = None):
        """
        Genera:
          - P_real_ext: (horizon+forecast_h,)
          - P_forecast_ext: (horizon+forecast_h,)
        y reinicia t=0.

        Returns:
          P_real (horizon,), P_forecast_ext (horizon+forecast_h,)
        """
        self._configure_rng(rng=rng, seed=seed)
        n_weeks = self.weeks_pool.shape[0]
        if week_idx is None:
            w = int(self._week_rng.integers(0, n_weeks))
        else:
            try:
                w = operator.index(week_idx)
            except TypeError as exc:
                raise TypeError("week_idx debe ser un índice entero de semana.") from exc
            if not 0 <= w < n_weeks:
                raise IndexError(f"week_idx={w} está fuera del pool [0, {n_weeks - 1}].")
        self.week_idx = w
        self.original_week_idx = int(self.valid_week_indices[w])

        # La semana histórica escalada es la base. Los atributos legacy de
        # week_noise se conservan por compatibilidad, pero ya no se aplican:
        # la única incertidumbre física es la realización 10 % desde la base.
        P_base = np.maximum(self.weeks_pool[w].astype(float), 0.0)
        pmax = float(np.max(P_base))
        self.scale = 1.0 if pmax < 1e-9 else (self.pmax_target_kw / pmax)
        self.P_base_profile = P_base * self.scale

        realization_sigma = self.realization_sigma_frac * np.abs(self.P_base_profile)
        self.realization_noise = self._realization_rng.normal(0.0, realization_sigma)
        # La demanda física no puede ser negativa. No se impone techo Pmax:
        # Pmax sólo normaliza/escala el perfil histórico base.
        self.P_real_profile = np.maximum(
            self.P_base_profile + self.realization_noise,
            0.0,
        )

        # Extensión cíclica sobre las mismas realizaciones ya muestreadas.
        self.P_base_ext = np.concatenate(
            [self.P_base_profile, self.P_base_profile[: self.forecast_h]], axis=0
        )
        P_real_ext = np.concatenate(
            [self.P_real_profile, self.P_real_profile[: self.forecast_h]], axis=0
        )

        # construye forecast sobre la serie extendida
        P_forecast_ext = self._make_forecast(P_real_ext)

        self.P_real_ext = P_real_ext
        self.P_forecast_ext = P_forecast_ext
        self.t = 0

        # El ambiente futuro debe usar value_at(t) y forecast_at(t), ambos
        # no mutantes. ``t`` queda únicamente para compatibilidad legacy.
        return self.P_real_ext[:self.horizon_hours], self.P_forecast_ext

    def step_value_and_forecast(self):
        """API legacy mutable; el ambiente nuevo no debe utilizarla.

        Conserva la semántica histórica de devolver ``[t, ..., t+10]`` y
        después incrementar ``self.t``. Para la interfaz oficial usa
        :meth:`value_at` y :meth:`forecast_at`.
        """
        self._require_prepared_profiles()
        t = self._validate_time_index(self.t)
        P_t = self.value_at(t)
        fc_11 = self.P_forecast_ext[t : t + self.forecast_h].astype(float, copy=True)

        self.t += 1
        done = (self.t >= self.horizon_hours)
        return P_t, fc_11, done
