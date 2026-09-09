import operator

import numpy as np


# Hipótesis de modelado local (no procedente del paper): mantiene una
# meteorología semanal común, pero evita perfiles reales idénticos entre PV.
PV_SPATIAL_RHO = 0.95
PV_SPATIAL_SIGMA = 0.02
PV_SPATIAL_CLIP = 0.05


class WeeklyPVSampler:
    """
    Sampler semanal para PV basado en un pool de semanas guardado en .npz.

    - Archivo .npz debe contener un array llamado 'weeks' con shape (n_weeks, 168).
    - En reset(), el perfil solar semanal muestreado es la meteorología base.
    - Se genera una realización meteorológica común a partir de la base y,
      después, la perturbación espacial AR(1) propia de cada planta.
    - El forecast se construye desde la potencia real final de cada planta.
    - Las consultas oficiales son no mutantes: ``value_at(t)`` y
      ``forecast_at(t)``; el forecast contiene ``[t+1, ..., t+11]``.
    """

    def __init__(
        self,
        filepath: str = "solar_weeks_pool.npz",
        seed: int = 0,
        pmax_target_kw: float | None = None,   # si None, no escala
        horizon_hours: int = 168,
        forecast_h: int = 11,
        realization_sigma_frac: float = 0.10,
        forecast_sigma_frac: float = 0.10,     # N(0, 0.1 * valor)
        pv_floor: float = 0.5,                 # offset nocturno: < pv_floor -> 0
        clip_nonnegative: bool = True,
        spatial_rho: float = PV_SPATIAL_RHO,
        spatial_sigma: float = PV_SPATIAL_SIGMA,
        spatial_clip: float = PV_SPATIAL_CLIP,
    ):
        self.filepath = filepath
        self.horizon_hours = int(horizon_hours)
        self.forecast_h = int(forecast_h)

        self.rng = np.random.default_rng(seed)

        self.pmax_target_kw = None if pmax_target_kw is None else float(pmax_target_kw)
        self.realization_sigma_frac = float(realization_sigma_frac)
        self.forecast_sigma_frac = float(forecast_sigma_frac)
        self.pv_floor = float(pv_floor)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.spatial_rho = float(spatial_rho)
        self.spatial_sigma = float(spatial_sigma)
        self.spatial_clip = float(spatial_clip)
        if not -1.0 < self.spatial_rho < 1.0:
            raise ValueError("spatial_rho debe estar estrictamente entre -1 y 1.")
        if self.spatial_sigma < 0.0:
            raise ValueError("spatial_sigma debe ser no negativa.")
        if self.spatial_clip < 0.0:
            raise ValueError("spatial_clip debe ser no negativo.")
        if self.realization_sigma_frac < 0.0 or not np.isfinite(self.realization_sigma_frac):
            raise ValueError("realization_sigma_frac debe ser finito y no negativo.")
        if self.forecast_sigma_frac < 0.0 or not np.isfinite(self.forecast_sigma_frac):
            raise ValueError("forecast_sigma_frac debe ser finito y no negativo.")

        self.weeks_pool = self._load_weeks_pool(filepath)

        # estado interno
        self.t = 0
        self.week_idx = None
        self.scale = 1.0
        self.common_base_weather = None   # meteorología base normalizada, len 168
        self.common_real_weather = None   # realización meteorológica común, len 168
        self.common_realization_noise = None
        self.P_base_week = None       # base por planta tras escalar, len 168
        self.P_common_real_week = None
        self.spatial_epsilon_week = None  # perturbación AR(1) multiplicativa, len 168
        self.P_real_week = None
        self.P_real_ext = None       # len = 168 + 11
        self.P_forecast_ext = None   # len = 168 + 11
        self.forecast_noise_ext = None
        self._week_rng = None
        self._realization_rng = None
        self._spatial_rng = None
        self._forecast_rng = None

    def _load_weeks_pool(self, filepath: str) -> np.ndarray:
        data = np.load(filepath, allow_pickle=True)
        if "weeks" not in data:
            raise KeyError(f"El archivo {filepath} no contiene la clave 'weeks'.")
        weeks = np.asarray(data["weeks"], dtype=float)
        if weeks.ndim != 2 or weeks.shape[1] != self.horizon_hours:
            raise ValueError(f"'weeks' debe tener shape (n_weeks, {self.horizon_hours}); tiene {weeks.shape}.")
        if weeks.shape[0] < 1:
            raise ValueError("El pool de semanas está vacío.")
        return weeks

    def _apply_floor(self, P: np.ndarray) -> np.ndarray:
        P = P.copy()
        P[np.isnan(P)] = 0.0
        P[P < self.pv_floor] = 0.0
        if self.clip_nonnegative:
            P = np.clip(P, 0.0, None)
        return P

    def base_weather_from_week(self, week_idx: int) -> np.ndarray:
        """Devuelve la meteorología base normalizada de una semana del pool."""
        try:
            idx = operator.index(week_idx)
        except TypeError as exc:
            raise TypeError("week_idx debe ser un índice entero de semana.") from exc
        if not 0 <= idx < self.weeks_pool.shape[0]:
            raise IndexError(
                f"week_idx={idx} está fuera del pool [0, {self.weeks_pool.shape[0] - 1}]."
            )
        week = self._apply_floor(np.asarray(self.weeks_pool[idx], dtype=float))
        peak = float(np.max(week))
        if peak < 1e-12:
            return np.zeros(self.horizon_hours, dtype=float)
        return week / peak

    @staticmethod
    def sample_common_real_weather(
        base_weather: np.ndarray,
        *,
        rng=None,
        seed: int | None = None,
        realization_sigma_frac: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Muestrea la capa meteorológica común sin generar PV nocturna."""
        if rng is not None and seed is not None:
            raise ValueError("Pasa rng o seed, no ambos.")
        if rng is None:
            rng = np.random.default_rng(seed)
        base = np.asarray(base_weather, dtype=float)
        if base.ndim != 1 or not np.all(np.isfinite(base)) or np.any(base < 0.0):
            raise ValueError("base_weather debe ser un vector finito y no negativo.")
        if realization_sigma_frac < 0.0 or not np.isfinite(realization_sigma_frac):
            raise ValueError("realization_sigma_frac debe ser finito y no negativo.")
        noise = rng.normal(0.0, realization_sigma_frac * np.abs(base)).astype(float)
        # Para base=0, sigma=0: la noche permanece exactamente en cero.
        return np.maximum(base + noise, 0.0), noise

    def _make_spatial_epsilon(self) -> np.ndarray:
        """Genera una perturbación AR(1) con sigma estacionaria aproximada.

        La innovación usa ``sigma * sqrt(1-rho**2)``; sin clipping, ello deja
        una desviación estándar estacionaria igual a ``spatial_sigma``.
        El recorte es una salvaguarda física adicional.
        """
        epsilon = np.empty(self.horizon_hours, dtype=float)
        epsilon[0] = self._spatial_rng.normal(0.0, self.spatial_sigma)
        innovation_sigma = self.spatial_sigma * np.sqrt(1.0 - self.spatial_rho**2)
        for t in range(1, self.horizon_hours):
            epsilon[t] = (
                self.spatial_rho * epsilon[t - 1]
                + self._spatial_rng.normal(0.0, innovation_sigma)
            )
        return np.clip(epsilon, -self.spatial_clip, self.spatial_clip)

    def _make_forecast_ext(self, P_real_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma = self.forecast_sigma_frac * np.abs(P_real_ext)
        noise = self._forecast_rng.normal(0.0, sigma)
        P_fc = P_real_ext + noise
        # El forecast es una predicción: no se recorta a [0, Pmax].
        return P_fc, noise

    def _configure_rng(self, *, rng=None, seed: int | None = None) -> None:
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

        week_seq, realization_seq, spatial_seq, forecast_seq = np.random.SeedSequence(entropy).spawn(4)
        self._week_rng = np.random.default_rng(week_seq)
        self._realization_rng = np.random.default_rng(realization_seq)
        self._spatial_rng = np.random.default_rng(spatial_seq)
        self._forecast_rng = np.random.default_rng(forecast_seq)

    def _require_prepared_profiles(self) -> None:
        if self.P_real_ext is None or self.P_forecast_ext is None:
            raise RuntimeError("Debes llamar reset() antes de consultar perfiles PV.")

    def _validate_time_index(self, t: int) -> int:
        try:
            t = operator.index(t)
        except TypeError as exc:
            raise TypeError("t debe ser un índice entero de hora.") from exc
        if not 0 <= t < self.horizon_hours:
            raise IndexError(f"t={t} está fuera del horizonte [0, {self.horizon_hours - 1}].")
        return int(t)

    def value_at(self, t: int) -> float:
        """Potencia PV disponible de la hora ``t`` sin cambiar estado."""
        self._require_prepared_profiles()
        t = self._validate_time_index(t)
        return float(self.P_real_ext[t])

    def forecast_at(self, t: int, horizon: int | None = None) -> np.ndarray:
        """Forecast no mutante para ``[t+1, ..., t+horizon]``.

        Cerca del final usa la extensión cíclica preparada en ``reset``. Es
        una aproximación de borde del episodio, no una semana futura observada.
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

    def reset(
        self,
        *,
        rng=None,
        seed: int | None = None,
        week_idx: int | None = None,
        common_base_weather: np.ndarray | None = None,
        common_real_weather: np.ndarray | None = None,
    ):
        self._configure_rng(rng=rng, seed=seed)
        # 1) samplear semana, o usar una semana meteorológica indicada por el
        # ambiente para mantener coherentes varias plantas del mismo sitio.
        if week_idx is None:
            idx = int(self._week_rng.integers(0, self.weeks_pool.shape[0]))
        else:
            try:
                idx = operator.index(week_idx)
            except TypeError as exc:
                raise TypeError("week_idx debe ser un índice entero de semana.") from exc
            if not 0 <= idx < self.weeks_pool.shape[0]:
                raise IndexError(
                    f"week_idx={idx} está fuera del pool [0, {self.weeks_pool.shape[0] - 1}]."
                )
        self.week_idx = idx
        own_base_weather = self.base_weather_from_week(idx)
        if common_base_weather is None:
            base_weather = own_base_weather
        else:
            base_weather = np.asarray(common_base_weather, dtype=float)
            if base_weather.shape != (self.horizon_hours,) or not np.all(np.isfinite(base_weather)):
                raise ValueError("common_base_weather debe ser finito y de longitud horizon_hours.")
            if np.any(base_weather < 0.0):
                raise ValueError("common_base_weather no puede contener valores negativos.")
            base_weather = base_weather.copy()

        if common_real_weather is None:
            common_real, common_noise = self.sample_common_real_weather(
                base_weather,
                rng=self._realization_rng,
                realization_sigma_frac=self.realization_sigma_frac,
            )
        else:
            common_real = np.asarray(common_real_weather, dtype=float)
            if common_real.shape != (self.horizon_hours,) or not np.all(np.isfinite(common_real)):
                raise ValueError("common_real_weather debe ser finito y de longitud horizon_hours.")
            if np.any(common_real < 0.0):
                raise ValueError("common_real_weather no puede contener valores negativos.")
            common_real = common_real.copy()
            common_noise = common_real - base_weather

        # 2) La base y la realización meteorológica común se convierten a la
        # escala de la planta. Para las plantas de la microrred, Pmax es el
        # límite físico final, no un recorte de la meteorología común.
        source_week = self._apply_floor(np.asarray(self.weeks_pool[idx], dtype=float))
        source_peak = float(np.max(source_week))
        if self.pmax_target_kw is not None:
            self.scale = 1.0 if source_peak < 1e-12 else (self.pmax_target_kw / source_peak)
            power_scale = self.pmax_target_kw
        else:
            self.scale = 1.0
            power_scale = source_peak
        self.common_base_weather = base_weather
        self.common_real_weather = common_real
        self.common_realization_noise = common_noise
        self.P_base_week = self.common_base_weather * power_scale
        self.P_common_real_week = self.common_real_weather * power_scale

        # 3) Diferencia local por planta: se aplica después de la capa común.
        epsilon = self._make_spatial_epsilon()
        if self.pmax_target_kw is not None:
            P_week = np.clip(
                self.P_common_real_week * (1.0 + epsilon), 0.0, self.pmax_target_kw
            )
        else:
            P_week = np.clip(self.P_common_real_week * (1.0 + epsilon), 0.0, None)
        self.spatial_epsilon_week = epsilon
        self.P_real_week = P_week

        # 4) Extensión +11h sobre las mismas realizaciones finales.
        ext = P_week[: self.forecast_h].copy()
        P_real_ext = np.concatenate([P_week, ext], axis=0)  # len 179

        # 5) Forecast sobre la potencia real final, con error propio.
        P_forecast_ext, forecast_noise_ext = self._make_forecast_ext(P_real_ext)

        self.P_real_ext = P_real_ext
        self.P_forecast_ext = P_forecast_ext
        self.forecast_noise_ext = forecast_noise_ext
        self.t = 0

        # por conveniencia devolvemos el vector real 168 y el forecast_ext 179
        return self.P_real_ext[: self.horizon_hours], self.P_forecast_ext

    def step_value_and_forecast(self):
        """API legacy mutable; el ambiente nuevo no debe utilizarla.

        Conserva ``[t, ..., t+10]`` para notebooks existentes. La interfaz
        oficial no mutante es :meth:`value_at` y :meth:`forecast_at`.
        """
        self._require_prepared_profiles()
        t = self._validate_time_index(self.t)
        P_t = self.value_at(t)
        fc_11 = self.P_forecast_ext[t : t + self.forecast_h].astype(float, copy=True)

        self.t += 1
        done = (self.t >= self.horizon_hours)
        return P_t, fc_11, done


class PVPlant:
    """
    Planta PV con acciones de:
      - curtailment u \in [0,1]  => P_inj = (1-u) * P_avail
      - Q_cmd_kvar (kVAr)        => saturado por S^2 >= P^2 + Q^2

    Nota práctica (modelado):
      - allow_statcom=True permite Q incluso si P_inj=0 (modo STATCOM)
      - allow_statcom=False fuerza Q=0 cuando P_inj ~ 0
    """

    def __init__(
        self,
        sampler: WeeklyPVSampler,
        s_max_kva: float,
        allow_statcom: bool = True,
    ):
        self.sampler = sampler
        self.s_max_kva = float(s_max_kva)
        self.allow_statcom = bool(allow_statcom)

        self.t = 0
        self.P_avail_week = None        # (168,)
        self.P_forecast_ext = None      # (179,)

        self.P_inj_kw = 0.0
        self.Q_inj_kvar = 0.0

    def _require_profiles(self) -> None:
        if self.P_avail_week is None or self.P_forecast_ext is None:
            raise RuntimeError("Debes llamar reset() antes de consultar o actuar sobre PVPlant.")

    def _validate_time_index(self, t: int) -> int:
        return self.sampler._validate_time_index(t)

    def reset(
        self,
        *,
        rng=None,
        seed: int | None = None,
        week_idx: int | None = None,
        common_base_weather: np.ndarray | None = None,
        common_real_weather: np.ndarray | None = None,
    ):
        P_week, fc_ext = self.sampler.reset(
            rng=rng,
            seed=seed,
            week_idx=week_idx,
            common_base_weather=common_base_weather,
            common_real_weather=common_real_weather,
        )
        self.P_avail_week = np.clip(np.asarray(P_week, float), 0.0, None)
        self.P_forecast_ext = np.asarray(fc_ext, float).copy()
        self.t = 0
        self.P_inj_kw = 0.0
        self.Q_inj_kvar = 0.0

    def available_at(self, t: int) -> float:
        """Consulta no mutante de disponibilidad PV en la hora externa ``t``."""
        self._require_profiles()
        return self.sampler.value_at(t)

    def forecast_at(self, t: int, horizon: int | None = None) -> np.ndarray:
        """Consulta no mutante del forecast futuro ``[t+1, ..., t+horizon]``."""
        self._require_profiles()
        return self.sampler.forecast_at(t, horizon=horizon)

    def get_available(self):
        """API legacy basada en ``self.t``; no usar desde el ambiente nuevo."""
        self._require_profiles()
        t = self._validate_time_index(self.t)
        P_t = self.available_at(t)
        fc_11 = self.P_forecast_ext[t : t + self.sampler.forecast_h].astype(float, copy=True)
        return P_t, fc_11

    def apply_action_at(self, t: int, curtailment: float, q_cmd_kvar: float):
        """
        Aplica la acción en la hora externa ``t`` sin avanzar ningún reloj.

        Inputs:
          curtailment: [0,1]
          q_cmd_kvar: setpoint de Q (kVAr)

        Outputs:
          (P_inj_kw, Q_inj_kvar) después de saturación
        """
        P_avail_t = self.available_at(t)

        u = float(np.clip(curtailment, 0.0, 1.0))
        P_inj = (1.0 - u) * P_avail_t

        if (not self.allow_statcom) and (P_inj <= 1e-6):
            Q_inj = 0.0
        else:
            # límite circular
            qmax = float(np.sqrt(max(self.s_max_kva**2 - P_inj**2, 0.0)))
            Q_inj = float(np.clip(q_cmd_kvar, -qmax, qmax))

        self.P_inj_kw = float(P_inj)
        self.Q_inj_kvar = float(Q_inj)
        return self.P_inj_kw, self.Q_inj_kvar

    def apply_resolved_setpoint_at(
        self,
        t: int,
        p_inj_kw: float,
        q_inj_kvar: float,
        *,
        available_kw: float | None = None,
    ):
        """Registra un setpoint PV ya resuelto sin volver a consultar el perfil.

        Esta ruta permite que un ambiente que aplica una disponibilidad efectiva
        (por ejemplo, un colapso de irradiancia) use exactamente el mismo P/Q
        tanto en el snapshot eléctrico como en el registro de la planta. La
        validación conserva los invariantes físicos, pero no recorta ni vuelve
        a resolver el setpoint proporcionado.
        """
        self._require_profiles()
        self._validate_time_index(t)
        try:
            p_inj = float(p_inj_kw)
            q_inj = float(q_inj_kvar)
        except (TypeError, ValueError) as exc:
            raise ValueError("Los setpoints PV resueltos deben ser numéricos.") from exc
        if not np.isfinite(p_inj) or not np.isfinite(q_inj):
            raise ValueError("Los setpoints PV resueltos deben ser finitos.")
        if p_inj < 0.0:
            raise ValueError("El setpoint activo PV no puede ser negativo.")
        if available_kw is not None:
            try:
                available = float(available_kw)
            except (TypeError, ValueError) as exc:
                raise ValueError("available_kw debe ser numérico.") from exc
            if not np.isfinite(available) or available < 0.0:
                raise ValueError("available_kw debe ser finito y no negativo.")
            if p_inj > available + 1e-9:
                raise ValueError("El setpoint activo PV supera la disponibilidad efectiva.")
        if (not self.allow_statcom) and p_inj <= 1e-6 and abs(q_inj) > 1e-9:
            raise ValueError("Esta planta no permite Q con inyección activa nula.")
        if p_inj**2 + q_inj**2 > self.s_max_kva**2 + 1e-8:
            raise ValueError("El setpoint PV resuelto supera la capability del inversor.")

        self.P_inj_kw = p_inj
        self.Q_inj_kvar = q_inj
        return self.P_inj_kw, self.Q_inj_kvar

    def apply_action(self, curtailment: float, q_cmd_kvar: float):
        """API legacy basada en ``self.t``; no usar desde el ambiente nuevo."""
        return self.apply_action_at(self.t, curtailment, q_cmd_kvar)

    def step_advance(self):
        """API legacy mutable; la autoridad temporal futura es el ambiente."""
        self._require_profiles()
        self._validate_time_index(self.t)
        self.t += 1
        done = (self.t >= len(self.P_avail_week))
        return done

    def get_last_injection(self):
        return self.P_inj_kw, self.Q_inj_kvar
