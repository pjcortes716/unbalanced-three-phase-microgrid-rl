# pyopendss.py
import opendssdirect as dss
import math
import pandas as pd
import uuid
from typing import List, Optional, Dict, Any


DEFAULT_LOAD_VMINPU = 0.80
DEFAULT_LOAD_VMAXPU = 1.10


class DSSCircuit:
    def __init__(
        self,
        name: str="circuit",
        basekv: float=12.47,
        phases: int=3,
        load_vminpu: float=DEFAULT_LOAD_VMINPU,
        load_vmaxpu: float=DEFAULT_LOAD_VMAXPU,
    ):
        self.name = name
        self.basekv = float(basekv)
        self.phases = int(phases)
        self.load_vminpu = float(load_vminpu)
        self.load_vmaxpu = float(load_vmaxpu)
        if not (
            math.isfinite(self.load_vminpu)
            and math.isfinite(self.load_vmaxpu)
            and 0.0 < self.load_vminpu <= self.load_vmaxpu
        ):
            raise ValueError("Se requiere 0 < load_vminpu <= load_vmaxpu y ambos finitos.")
        self.vsource_name = "Source"
        self.pcc_bus = None
        # dss.Command + excepciones permite detectar comandos DSS inválidos antes
        # de que OpenDSS aplique valores por defecto silenciosamente.
        dss.Error.UseExceptions(True)
        # limpiar y crear circuito
        self._run_checked("Clear")
        self._run_checked(
            f"New Circuit.{self.name} basekv={self.basekv} pu=1.0 phases={self.phases}"
        )
        # forzar default sensible
        self._run_checked("Set DefaultBaseFrequency=60")
        # Mantener el flujo de potencia: ``LoadModel=1`` equivale a Admittance
        # en OpenDSS y haría que las cargas variaran aproximadamente con V^2.
        self._run_checked("Set LoadModel=PowerFlow")
        # Parámetros numéricos, no eléctricos: una tolerancia estricta permite
        # cerrar el balance de potencia cuando se incluye el transformador.
        self._run_checked("Set MaxIterations=500")
        self._run_checked("Set Tol=1e-9")

    def _run_checked(self, command: str):
        """Ejecuta un comando DSS y propaga errores de sintaxis o de modelo."""
        try:
            dss.Command(command)
        except Exception as exc:
            raise RuntimeError(f"OpenDSS rechazó el comando: {command}") from exc

        error_number = int(dss.Error.Number())
        if error_number:
            description = str(dss.Error.Description())
            raise RuntimeError(
                f"OpenDSS reportó error #{error_number} para '{command}': {description}"
            )

    def clear(self):
        self._run_checked("Clear")

    def connect_to_grid(self, bus: str="bus1", pu: float=1.0, angle: float=0.0, vsrc_name: str="Source"):
        """Configura la única fuente creada automáticamente por ``New Circuit``."""
        bus = str(bus).lower()
        if str(vsrc_name).lower() != "source":
            raise ValueError(
                "New Circuit crea Vsource.Source; no se permite crear una segunda "
                "Vsource mediante connect_to_grid(). Use vsrc_name='Source'."
            )
        self.vsource_name = "Source"
        self.pcc_bus = bus
        self._run_checked(
            f"Edit Vsource.Source bus1={bus} phases={self.phases} "
            f"pu={float(pu)} basekv={self.basekv} angle={float(angle)}"
        )

    def set_voltage_bases(self, voltage_bases_kv_ll: List[float]) -> None:
        """Configura bases L-L y recalcula las bases L-N de cada barra.

        ``DSSCircuit.basekv`` conserva la base de la fuente de este circuito,
        pero un circuito con transformadores requiere que OpenDSS determine la
        base propia de cada nivel de tensión. Los valores recibidos se expresan
        en kV L-L, tal como espera ``VoltageBases``.
        """
        values = [float(value) for value in voltage_bases_kv_ll]
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("voltage_bases_kv_ll debe contener bases L-L positivas y finitas.")
        formatted = ", ".join(f"{value:.12g}" for value in values)
        self._run_checked(f"Set VoltageBases=[{formatted}]")
        self._run_checked("CalcVoltageBases")

    def add_transformer(
        self,
        name: str,
        *,
        primary_bus: str,
        secondary_bus: str,
        phases: int,
        kva: float,
        primary_kv: float,
        secondary_kv: float,
        r_percent_total: float,
        xhl_percent: float,
        primary_connection: str = "wye",
        secondary_connection: str = "wye",
        primary_rneut_ohm: float = -1.0,
        primary_xneut_ohm: float = 0.0,
        secondary_rneut_ohm: float = -1.0,
        secondary_xneut_ohm: float = 0.0,
    ) -> str:
        """Añade un transformador de dos devanados con parámetros explícitos.

        ``r_percent_total`` es la resistencia serie total: se distribuye por
        igual entre ambos devanados para que OpenDSS represente ese total.
        ``xhl_percent`` es la reactancia serie entre devanados (``XHL``).
        Los neutros se declaran explícitamente incluso cuando ``Rneut=-1``;
        esto evita que la representación dependa de valores por defecto.
        """
        transformer_name = str(name).strip()
        if not transformer_name:
            raise ValueError("name del transformador no puede estar vacío.")
        if int(phases) <= 0:
            raise ValueError("phases debe ser positivo.")
        numeric_values = {
            "kva": kva,
            "primary_kv": primary_kv,
            "secondary_kv": secondary_kv,
            "r_percent_total": r_percent_total,
            "xhl_percent": xhl_percent,
            "primary_rneut_ohm": primary_rneut_ohm,
            "primary_xneut_ohm": primary_xneut_ohm,
            "secondary_rneut_ohm": secondary_rneut_ohm,
            "secondary_xneut_ohm": secondary_xneut_ohm,
        }
        if any(not math.isfinite(float(value)) for value in numeric_values.values()):
            raise ValueError("Los parámetros del transformador deben ser finitos.")
        if float(kva) <= 0.0 or float(primary_kv) <= 0.0 or float(secondary_kv) <= 0.0:
            raise ValueError("kva, primary_kv y secondary_kv deben ser positivos.")
        if float(r_percent_total) < 0.0 or float(xhl_percent) < 0.0:
            raise ValueError("r_percent_total y xhl_percent no pueden ser negativos.")

        allowed_connections = {"wye", "delta"}
        primary_connection = str(primary_connection).lower()
        secondary_connection = str(secondary_connection).lower()
        if primary_connection not in allowed_connections or secondary_connection not in allowed_connections:
            raise ValueError("Las conexiones del transformador deben ser 'wye' o 'delta'.")

        winding_r_percent = float(r_percent_total) / 2.0
        self._run_checked(
            f"New Transformer.{transformer_name} phases={int(phases)} windings=2 "
            f"XHL={float(xhl_percent):.12g}"
        )
        self._run_checked(
            f"~ wdg=1 bus={str(primary_bus).lower()} conn={primary_connection} "
            f"kv={float(primary_kv):.12g} kva={float(kva):.12g} %R={winding_r_percent:.12g}"
        )
        self._run_checked(
            f"~ wdg=1 Rneut={float(primary_rneut_ohm):.12g} "
            f"Xneut={float(primary_xneut_ohm):.12g}"
        )
        self._run_checked(
            f"~ wdg=2 bus={str(secondary_bus).lower()} conn={secondary_connection} "
            f"kv={float(secondary_kv):.12g} kva={float(kva):.12g} %R={winding_r_percent:.12g}"
        )
        self._run_checked(
            f"~ wdg=2 Rneut={float(secondary_rneut_ohm):.12g} "
            f"Xneut={float(secondary_xneut_ohm):.12g}"
        )
        return f"Transformer.{transformer_name}"

    def solve(self, mode: str='snap'):
        self._run_checked(f"Set mode={mode}")
        self._run_checked("Solve")
        return self.is_converged()

    def is_converged(self) -> bool:
        return bool(dss.Solution.Converged())

    # ---------- lines ----------
    @staticmethod
    def _matrix_to_dss(matrix: List[List[float]]) -> str:
        """Formatea la triangular inferior que espera RMatrix/XMatrix de OpenDSS."""
        return "(" + " | ".join(
            " ".join(f"{float(matrix[row][col]):.12g}" for col in range(row + 1))
            for row in range(3)
        ) + ")"

    @staticmethod
    def _per_phase_to_diagonal_matrix(values: List[float], name: str) -> List[List[float]]:
        if len(values) != 3:
            raise ValueError(f"{name} debe contener exactamente tres valores.")
        return [
            [float(values[0]), 0.0, 0.0],
            [0.0, float(values[1]), 0.0],
            [0.0, 0.0, float(values[2])],
        ]

    def add_unbalanced_line(self, line_name: str, from_bus: str, to_bus: str,
                             length: float = 1.0, units: str = "km",
                             r_per_phase: Optional[List[float]] = None,
                             x_per_phase: Optional[List[float]] = None,
                             z_matrix: Optional[List[List[float]]] = None,
                             c1: float = 0.0, basefreq: int = 60) -> Dict[str,str]:
        """Añade una línea trifásica mediante matrices R/X en ohm/km.

        Cuando se pasan R/X por fase, se usan matrices diagonales: el modelo no
        inventa acoplamientos mutuos, secuencia cero ni parámetros de neutro.
        """
        from_bus = str(from_bus).lower()
        to_bus = str(to_bus).lower()
        lc_name = f"LC_{line_name}_{uuid.uuid4().hex[:6]}"
        if z_matrix is not None:
            if len(z_matrix) != 3 or any(len(row) != 3 for row in z_matrix):
                raise ValueError("z_matrix debe ser una matriz 3x3.")
            r_matrix = [[float(complex(value).real) for value in row] for row in z_matrix]
            x_matrix = [[float(complex(value).imag) for value in row] for row in z_matrix]
        else:
            if r_per_phase is None or x_per_phase is None:
                raise ValueError("r_per_phase and x_per_phase required if no z_matrix")
            r_matrix = self._per_phase_to_diagonal_matrix(r_per_phase, "r_per_phase")
            x_matrix = self._per_phase_to_diagonal_matrix(x_per_phase, "x_per_phase")

        if float(c1) != 0.0:
            raise ValueError(
                "c1 no es compatible con el modelo matricial R/X de esta microrred; "
                "no se definen parámetros de secuencia."
            )

        unit = str(units).lower()
        if unit == "km":
            length_km = float(length)
        elif unit == "m":
            length_km = float(length) / 1000.0
        else:
            raise ValueError("units debe ser 'km' o 'm' para impedancias expresadas en ohm/km.")

        self._run_checked(
            f"New LineCode.{lc_name} nphases=3 units=km BaseFreq={int(basefreq)} "
            f"RMatrix={self._matrix_to_dss(r_matrix)} "
            f"XMatrix={self._matrix_to_dss(x_matrix)}"
        )
        self._run_checked(
            f"New Line.{line_name} bus1={from_bus} bus2={to_bus} phases=3 "
            f"length={length_km:.12g} units=km LineCode={lc_name}"
        )
        return {"line_name": line_name, "linecode": lc_name}

    # ---------- loads ----------
    def _kv_ln_of_bus(self, bus: str, kv_ll: Optional[float] = None) -> float:
        """Devuelve kV L-N a partir de una base L-L explícita o del circuito."""
        base_kv_ll = self.basekv if kv_ll is None else float(kv_ll)
        if not math.isfinite(base_kv_ll) or base_kv_ll <= 0.0:
            raise ValueError("kv_ll debe ser positivo y finito.")
        return base_kv_ll / math.sqrt(3.0)

    def add_load(self, name: str, bus: str, phases: int=1, kw: float=0.0, kvar: float=0.0, pf: Optional[float]=None, model: int=1, kv_ll: Optional[float]=None, vminpu: Optional[float]=None, vmaxpu: Optional[float]=None):
        """
        Create load. By default model=1 (constant P) to guarantee resolved power == assigned.
        For monophase use bus like 'bus1.1' OR pass bus='bus1' and phases=1 will attach to bus.1 automatically.
        """
        bus = str(bus).lower()
        # Modelo 1 con límites de tensión explícitos: evita que OpenDSS cambie
        # silenciosamente a la característica de baja/alta tensión por defecto.
        vminpu = self.load_vminpu if vminpu is None else float(vminpu)
        vmaxpu = self.load_vmaxpu if vmaxpu is None else float(vmaxpu)
        if not (math.isfinite(vminpu) and math.isfinite(vmaxpu) and 0.0 < vminpu <= vmaxpu):
            raise ValueError("Se requiere 0 < vminpu <= vmaxpu y ambos finitos.")
        if phases == 1:
            if '.' in bus:
                bus_spec = bus
            else:
                bus_spec = f"{bus}.1"
            kv = self._kv_ln_of_bus(bus, kv_ll=kv_ll)
        else:
            bus_spec = bus
            kv = self.basekv if kv_ll is None else float(kv_ll)

        if pf is not None and kvar == 0.0:
            reactive_spec = f"pf={float(pf)}"
        else:
            reactive_spec = f"kvar={float(kvar)}"

        conn_spec = " conn=wye" if phases == 1 else ""
        self._run_checked(
            f"New Load.{name} bus1={bus_spec} phases={int(phases)} kv={float(kv):.12g} "
            f"kW={float(kw)} {reactive_spec} model={int(model)} "
            f"Vminpu={vminpu:.12g} Vmaxpu={vmaxpu:.12g}{conn_spec}"
        )

    def add_fixed_injection(self, name: str, bus: str, phases: int = 1, kw: float = 0.0, kvar: float = 0.0, kv_ll: Optional[float] = None, vminpu: Optional[float] = None, vmaxpu: Optional[float] = None):
        """
        Inyección/absorción determinista implementada con una ``Load`` negativa.

        Convención física de esta interfaz: kw/kvar positivos inyectan en la
        barra; valores negativos absorben. En OpenDSS se asignan -kw y -kvar.
        """
        bus = str(bus).lower()
        vminpu = self.load_vminpu if vminpu is None else float(vminpu)
        vmaxpu = self.load_vmaxpu if vmaxpu is None else float(vmaxpu)
        if not (math.isfinite(vminpu) and math.isfinite(vmaxpu) and 0.0 < vminpu <= vmaxpu):
            raise ValueError("Se requiere 0 < vminpu <= vmaxpu y ambos finitos.")
        if phases == 1:
            bus_spec = bus if "." in bus else f"{bus}.1"
            kv = self._kv_ln_of_bus(bus, kv_ll=kv_ll)
        else:
            bus_spec = bus
            kv = self.basekv if kv_ll is None else float(kv_ll)
        conn_spec = " conn=wye" if phases == 1 else ""
        self._run_checked(
            f"New Load.{name} bus1={bus_spec} phases={int(phases)} kv={float(kv):.12g} "
            f"kW={-float(kw)} kvar={-float(kvar)} model=1 "
            f"Vminpu={vminpu:.12g} Vmaxpu={vmaxpu:.12g}{conn_spec}"
        )
        return f"Load.{name}"

    # ---------- generators ----------
    def add_generator(self, name: str, bus: str, phases: int=3, kw: float=0.0, kvar: float=0.0, kva: Optional[float]=None, enforce_limits: bool=False, kv_ll: Optional[float]=None):
        """
        Add generator. If enforce_limits=True, sets kWmax=kWmin=kw to try to fix output.
        For absolute determinism use add_fixed_injection (load negative).
        """
        bus = str(bus).lower()
        kv = self._kv_ln_of_bus(bus, kv_ll=kv_ll) if phases == 1 else (self.basekv if kv_ll is None else float(kv_ll))
        cmd = (
            f"New Generator.{name} bus1={bus} phases={int(phases)} "
            f"kv={float(kv):.12g} kW={float(kw)} kvar={float(kvar)}"
        )
        if kva is not None:
            cmd += f' kVA={kva}'
        if enforce_limits:
            cmd += f' kWmax={kw} kWmin={kw}'
        self._run_checked(cmd)
        self._run_checked("Solve")
        return f"Generator.{name}"

    def edit(self, element_fullname: str, **props):
        kvs = " ".join([f"{k}={v}" for k,v in props.items()])
        self._run_checked(f'Edit {element_fullname} {kvs}')

    # ---------- inspections & helpers ----------
    def _parse_pvars(self, pvars: List[float]):
        """Convierte ``CktElement.Powers()`` nativo de OpenDSS a pares kW/kvar."""
        vals = [float(x) for x in pvars]
        units = "kW/kvar"
        pairs = [(vals[i], vals[i+1] if i+1 < len(vals) else 0.0) for i in range(0, len(vals), 2)]
        P_total = sum(p for p,q in pairs)
        Q_total = sum(q for p,q in pairs)
        return pairs, P_total, Q_total, units

    def _terminal_power_pairs(self, element_fullname: str, terminal_index: int = 0):
        """Devuelve los pares P/Q nativos de OpenDSS para un terminal concreto.

        En ``CktElement.Powers()``, P/Q positivos significan potencia entrando al
        elemento activo por dicho terminal. La conversión a la convención física
        del PCC se realiza solamente en ``substation_phase_powers_fixed``.
        """
        dss.Circuit.SetActiveElement(element_fullname)
        nterm = int(dss.CktElement.NumTerminals())
        ncond = int(dss.CktElement.NumConductors())
        if terminal_index < 0 or terminal_index >= nterm:
            raise ValueError(
                f"El terminal {terminal_index} no existe en {element_fullname}; "
                f"el elemento tiene {nterm} terminal(es)."
            )

        pvars = [float(value) for value in (dss.CktElement.Powers() or [])]
        values_per_terminal = 2 * ncond
        start = terminal_index * values_per_terminal
        end = start + values_per_terminal
        if len(pvars) < end:
            raise RuntimeError(
                f"OpenDSS devolvió {len(pvars)} valores de potencia para "
                f"{element_fullname}; se esperaban al menos {end}."
            )
        terminal_values = pvars[start:end]
        return [
            (terminal_values[i], terminal_values[i + 1])
            for i in range(0, len(terminal_values), 2)
        ]

    def measure_power(self, element_fullname: str):
        """Mide P/Q con la convención nativa de ``CktElement.Powers()``."""
        dss.Circuit.SetActiveElement(element_fullname)
        pvars = dss.CktElement.Powers() or []
        pairs, P_total, Q_total, units = self._parse_pvars(pvars)
        return {"element": element_fullname, "pairs": pairs, "P_kW": P_total, "Q_kvar": Q_total, "units": units, "raw": pvars}

    def measure_power_via_powercmd(self, element_short: str):
        """Run the textual 'Power <element>' command and parse result (useful for Vsource)."""
        txt = dss.run_command(f"Power {element_short}")
        vals = [float(x) for x in txt.replace(',', ' ').split() if x.strip()!='']
        pairs = [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
        P_total = sum(p for p,q in pairs)
        Q_total = sum(q for p,q in pairs)
        return {"element": element_short, "pairs": pairs, "P_kW": P_total, "Q_kvar": Q_total}

    def get_bus_voltages_df(self, buses: Optional[List[str]] = None) -> pd.DataFrame:
        """Tensiones de barra por fase, incluyendo ``V_pu`` sobre base L-N."""
        rows = []
        if buses is None:
            buses = dss.Circuit.AllBusNames()
        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            v = dss.Bus.Voltages()
            nodes = dss.Bus.Nodes()
            if not v:
                continue
            kv_base_ln = float(dss.Bus.kVBase())
            # Los circuitos LV históricos no llamaban CalcVoltageBases(). Se
            # conserva su base como respaldo, pero en redes multinivel se usa
            # siempre la base específica calculada por OpenDSS para cada bus.
            if not math.isfinite(kv_base_ln) or kv_base_ln <= 0.0:
                kv_base_ln = self._kv_ln_of_bus(str(bus))
            v_base_ln_v = kv_base_ln * 1000.0
            for i,node in enumerate(nodes):
                Vr = v[2*i]; Vi = v[2*i+1]
                rows.append({
                    "bus": bus,
                    "phase": int(node),
                    "V_mag_V": math.hypot(Vr,Vi),
                    "V_ang_deg": math.degrees(math.atan2(Vi,Vr)),
                    "V_pu": math.hypot(Vr,Vi) / v_base_ln_v,
                })
        return pd.DataFrame(rows)

    def list_loads_df(self) -> pd.DataFrame:
        rows = []
        idx = dss.Loads.First()
        while idx > 0:
            name = dss.Loads.Name()
            fullname = name if name.lower().startswith("load.") else f"Load.{name}"
            dss.Circuit.SetActiveElement(fullname)
            buses = dss.CktElement.BusNames()
            if isinstance(buses,str): buses=[buses]
            assigned_kw = dss.Loads.kW()
            assigned_kvar = dss.Loads.kvar()
            pvars = dss.CktElement.Powers() or []
            pairs, P_total, Q_total, units = self._parse_pvars(pvars)
            rows.append({"element":fullname, "buses":",".join(buses), "assigned_kW":assigned_kw, "assigned_kvar":assigned_kvar, "resolved_kW":P_total, "resolved_kvar":Q_total, "units":units})
            idx = dss.Loads.Next()
        return pd.DataFrame(rows)

    def get_power_balance(self):
        """Balance físico en la convención de la microrred.

        ``P_grid_import_kW`` y ``Q_grid_import_kvar`` son positivos al importar
        desde la red externa. Las inyecciones de ``Generator`` son positivas y
        las ``Load`` (incluidas cargas negativas) se agregan como consumo neto.
        Por tanto, el residual esperado es::

            P_grid_import + P_generators_inj - P_loads_net - P_losses ~= 0
        """
        pcc_powers, _ = self.substation_phase_powers_fixed()
        pcc_total = pcc_powers[pcc_powers["phase"] == "TOTAL"].iloc[0]

        loads_df = self.list_loads_df()
        p_loads = float(loads_df["resolved_kW"].sum()) if not loads_df.empty else 0.0
        q_loads = float(loads_df["resolved_kvar"].sum()) if not loads_df.empty else 0.0

        p_generators_inj = 0.0
        q_generators_inj = 0.0
        for generator_name in dss.Generators.AllNames() or []:
            pairs = self._terminal_power_pairs(f"Generator.{generator_name}")
            p_generators_inj -= sum(p for p, _ in pairs)
            q_generators_inj -= sum(q for _, q in pairs)

        losses = dss.Circuit.Losses() or [0.0,0.0]
        p_losses = float(losses[0]) / 1000.0
        q_losses = float(losses[1]) / 1000.0
        p_grid_import = float(pcc_total["P_kW"])
        q_grid_import = float(pcc_total["Q_kvar"])

        return {
            "P_grid_import_kW": p_grid_import,
            "Q_grid_import_kvar": q_grid_import,
            "P_generators_inj_kW": p_generators_inj,
            "Q_generators_inj_kvar": q_generators_inj,
            "P_loads_net_kW": p_loads,
            "Q_loads_net_kvar": q_loads,
            "P_losses_kW": p_losses,
            "Q_losses_kvar": q_losses,
            "balance_residual_kW": p_grid_import + p_generators_inj - p_loads - p_losses,
            "balance_residual_kvar": q_grid_import + q_generators_inj - q_loads - q_losses,
            "loads_df": loads_df,
        }
    
    def substation_phase_powers_fixed(self, substation_bus: Optional[str] = None):
        """Mide el PCC de la única Vsource con la convención física del ambiente.

        ``P_grid > 0``/``Q_grid > 0`` significan importación desde la red
        externa. OpenDSS reporta el signo contrario en el terminal 1 de la
        ``Vsource`` porque allí la fuente entrega potencia al circuito.
        """
        source_element = f"Vsource.{self.vsource_name}"
        source_names = [name.lower() for name in (dss.Vsources.AllNames() or [])]
        if source_element.split(".", 1)[1].lower() not in source_names:
            raise RuntimeError(f"No existe la fuente efectiva {source_element}.")
        if len(source_names) != 1:
            raise RuntimeError(
                f"El PCC requiere una única Vsource; OpenDSS contiene {source_names}."
            )

        dss.Circuit.SetActiveElement(source_element)
        bus_names = list(dss.CktElement.BusNames() or [])
        if not bus_names:
            raise RuntimeError(f"{source_element} no tiene buses configurados.")
        source_bus = str(bus_names[0]).split(".")[0].lower()
        target_bus = source_bus if substation_bus is None else str(substation_bus).lower()
        if target_bus != source_bus:
            raise ValueError(
                f"El PCC de {source_element} está en '{source_bus}', no en '{target_bus}'."
            )

        native_pairs = self._terminal_power_pairs(source_element, terminal_index=0)
        nphases = int(dss.CktElement.NumPhases())
        if len(native_pairs) < nphases:
            raise RuntimeError(
                f"{source_element} devolvió {len(native_pairs)} pares P/Q para {nphases} fases."
            )

        rows = [
            {
                "element": source_element,
                "bus": source_bus,
                "phase": phase + 1,
                "P_kW": -float(native_pairs[phase][0]),
                "Q_kvar": -float(native_pairs[phase][1]),
            }
            for phase in range(nphases)
        ]
        phase_df = pd.DataFrame(rows)
        total = pd.DataFrame([{
            "element": source_element,
            "bus": source_bus,
            "phase": "TOTAL",
            "P_kW": float(phase_df["P_kW"].sum()),
            "Q_kvar": float(phase_df["Q_kvar"].sum()),
        }])
        debug_df = pd.DataFrame([{
            "element": source_element,
            "pcc_bus": source_bus,
            "native_terminal": 0,
            "native_sign": "P/Q positivos entran a la Vsource",
        }])
        return pd.concat([phase_df, total], ignore_index=True, sort=False), debug_df

