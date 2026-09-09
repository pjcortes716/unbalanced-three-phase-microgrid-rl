> Distribution note (2026-09-09): historical technical document, with personal paths sanitized. The original audit is subordinate to BENCHMARK_V1_CHANGELOG.md. References to excluded artifacts are historical. See the snapshot addendum in the changelog for current packaging differences.

# Auditoría técnica del ambiente trifásico
Fecha: 8 de septiembre de 2026. Revisión auditada: `863e699941517ea632cc93a22b24e9802998a39b` (31-08-2026).

Alcance: repositorio **<LOCAL_PATH>`-B`, sin guardar scripts ni resultados auxiliares.

Convenciones de evidencia: **E:línea** identifica [Ambiente_industrial.py](<env/Ambiente_industrial.py>); **D:línea**, [opendss_easy.py](<env/opendss_easy.py>); **B**, [battery.py](<env/battery.py>); **T**, [v2g_bus_5.py](<env/v2g_bus_5.py>); **Z**, [vehiculos_Zoe.py](<env/vehiculos_Zoe.py>); **L**, [oficinas.py](<env/oficinas.py>); **PV**, [PV_plant.py](<env/PV_plant.py>); **P**, [price_sampler.py](<env/price_sampler.py>); **X**, [extreme_events.py](<env/extreme_events.py>). Los rangos son líneas del código auditado, no comentarios tomados como prueba. **S1–S6** son verificaciones ejecutadas y descritas en §10. Las propuestas se identifican como tales.

## 1. Mapa general del proyecto

| Componente | Archivo(s) | Función/clase principal | Estado |
| --- | --- | --- | --- |
| Ambiente y API Gymnasium | E | MicrorredIndustrial, reset, step | Funcional; observación 114D, acción 12D; 43 pruebas seleccionadas pasan |
| Reward | E:147–217, 1638–1662 | compute_reward, RewardBreakdown | Costo económico + penalización fija PF |
| Solver y conexión Python | D | DSSCircuit | OpenDSSDirect.py / DSS C-API, contexto global |
| Red eléctrica | E:2001–2107, D:100–320 | build_industrial_network_with_transformer | Generada con comandos Python; no archivos .dss en el repositorio inspeccionado |
| Cargas | L; pools office/warehouse1/warehouse2 | OfficeLoadSampler | Semanas reales reescaladas, realizaciones y forecasts sintéticos |
| PV | PV; solar_weeks_pool.npz | WeeklyPVSampler, PVPlant | Tres plantas; meteorología común y variación espacial |
| Precios | P | PriceSampler | Perfil sintético diario, ruido y misma tarifa compra/venta |
| BESS | B | Battery, BatteryParams | Dinámica energética y saturación P/Q/SOC |
| Camiones | T | V2GTruck | Dos V2G, horarios fijos, SOC de llegada aleatorio |
| EV de oficina | Z | ZoeFleetBus3 | Dos Zoe exógenos, carga monofásica en A/B |
| Generadores convencionales | D:325–342 | add_generator | Helper disponible; no usado por step ni recursos térmicos en producción |
| Eventos extremos | X; E:930–1044 | ExtremeEventManager | load_surge, phase_imbalance, pv_collapse, compound, random |
| Wrappers / registro Gym | Búsqueda en fuentes y celdas de notebooks | — | No wrappers de entrenamiento, Monitor, VecNormalize ni registro gym.make |
| Normalización | E:838–872, 1271–1315 | _normalize_obs | Escalas físicas fijas; no estadísticas aprendidas |
| Logging / render | E:1437–1635, 1880–1937 | info, render, _record_render_step | Telemetría por paso; dashboard opcional en memoria; no logger RL persistente |
| Pruebas | `carpeta tests` (historical reference; not distributed) | unittest, diagnósticos eléctricos | 21 archivos test_*.py; se ejecutó una selección breve de 43 tests |
| Evaluación previa | tests/transformer_episode_diagnostics.py, stress_voltage_analysis.py, vminpu_experiment.py | _run_campaign y snapshots | Campañas eléctricas; algunas no utilizan env.step |
| Entrenamiento / SAC / Lagrangiano | Inventario completo de .py y celdas de código .ipynb | — | No encontrados |
| Modelos / replay buffers / Optuna | Inventario .zip/.pt/.pth/.pkl/.db/.sqlite y búsqueda de símbolos | — | Ninguno de RL; el ZIP encontrado es el dataset BDG2 |
| Resultados existentes | tests/*results.json, vminpu_experiment_before.json | Diccionarios diagnósticos | Resultados eléctricos, no curvas de aprendizaje ni agentes evaluados |

Conexión efectiva: reset selecciona/genera perfiles; step decodifica controles y resuelve factibilidad local de DER; construye el snapshot eléctrico; obtiene PCC/tensiones/pérdidas; calcula reward; confirma energías y avanza la hora. La topología de producción incluye transformador. El helper histórico build_industrial_ring_400V conserva otra red con fuente en bus2: no confundir sus resultados con los de step (E:1825, 2041–2085).

La búsqueda incluyó los notebooks de CODIGO y del dataset, examinando las celdas de código, no solo nombres ni imágenes embebidas. Las coincidencias de “sac” dentro de “transaccional” no son algoritmos SAC. No se afirma nada sobre otros repositorios o tareas fuera de este árbol.

## 2. Ambiente exacto y ciclo temporal

**Clase:** MicrorredIndustrial(gym.Env), E:410. Constructor público (E:524–532):

```python
from env.Ambiente_industrial import MicrorredIndustrial
env = MicrorredIndustrial(
    lambda_pf=1.0,
    extreme_events=None,
    extreme_event_seed=None,
    render_mode=None,
)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

| Parámetro | Valor/efecto |
| --- | --- |
| seed del constructor | Inicializa RNG/componentes, pero NO sustituye reset(seed=...) de Gym; S4 comprueba que constructor(seed=42).reset() no se reproduce |
| lambda_pf | Escalar finito ≥0; por defecto 1 USD/(kvar·h), fijo durante el episodio |
| extreme_events | None, “none”, tipo concreto, “random” o ExtremeEventSpec |
| extreme_event_seed | Semilla independiente opcional; la explícita repite el programa de eventos |
| render_mode | None o “episode”; None evita guardar historial de render |
| options en reset | Aceptado pero ignorado: no permite elegir split, semana ni SOC |
| Horizonte | 168 pasos de 1 h = una semana; forecast de 11 h |
| Final natural | Después de actuar en t=167: terminated=True, truncated=False |
| Truncamiento | Nunca True en la implementación |
| Fallo eléctrico | Excepción, no transición terminal ni truncada |
| Observación terminal | 114 ceros, sentinel explícito en info; no es el SOC físico final |
| Uso inválido | step antes de reset o después del final levanta RuntimeError |

Los tamaños y dt son atributos internos fijos, no parámetros del constructor (E:578–580). Cambiar solamente dt_h rompería la coherencia con vehículos, cuya integración está codificada en horas; no es una capacidad configurable validada.

Pseudocódigo fiel (E:1318–1421, 1762–1947):

```text
reset(seed):
    Gym.reset(seed); t=0; limpiar flags, acción e historial opcional
    derivar 9 semillas desde np_random; preparar precios
    elegir semanas de office/wh1/wh2
    elegir UNA semana solar y realización meteorológica común
    preparar las 3 plantas con perturbaciones locales y forecasts
    BESS SOC=0.50; reiniciar trucks y Zoe
    muestrear PF de edificios y reparto ABC; preparar evento opcional
    preparar llegadas/Zoe de t=0 y datos efectivos de hora 0
    devolver observación normalizada e info (SIN flujo de potencia)

step(a_t):
    validar episodio; preparar estado horario t idempotentemente
    validar 12 valores finitos y clip [-1,1]
    decodificar PV; resolver P/Q/SOC de BESS y trucks sin confirmar energía
    crear red desde cero; CalcVoltageBases; añadir cargas e inyecciones
    Solve; comprobar convergencia, tensiones y datos PCC
    si falla: excepción ANTES del commit energético
    calcular r_t usando precio_t y PCC_t
    confirmar PV/BESS/trucks; integrar Zoe una vez
    avanzar trucks; sus déficits de SOC de salida se calculan y se descartan
    construir info de la transición t -> t+1
    t=t+1
    si t==168: devolver sentinel, r_t, True, False, info
    preparar observación de t+1; devolver obs, r_t, False, False, info
```

No existe una fase separada de “verificar seguridad de red y corregir acción”. Las restricciones locales se aplican antes del solver; el PF se penaliza después; los voltajes/VUF/sobrecarga se reportan.

## 3. Topología y modelo eléctrico

**Sí es una red trifásica con solución explícita desbalanceada.** S1 verificó 6 buses, 18 nodos eléctricos de fase y 22 elementos: 1 fuente + 1 transformador + 5 líneas + 15 Load. Cada bus tiene nodos 1/2/3. No hay conductor neutro explícito distribuido.

| Bus | Base nominal L-L / L-N | Fases | Recursos efectivos |
| --- | --- | --- | --- |
| bus1 | 15 kV / 8.660254 kV | ABC | Única Vsource.Source; PCC de medición; primario Tx |
| bus2 | 0.4 kV / 0.230940 kV | ABC | Secundario Tx y entrada al anillo |
| bus3 | 0.4 kV / 0.230940 kV | ABC | Oficina + Zoe A/B; PV3 |
| bus4 | 0.4 kV / 0.230940 kV | ABC | BESS |
| bus5 | 0.4 kV / 0.230940 kV | ABC | Warehouse1, PV1, Truck5 |
| bus6 | 0.4 kV / 0.230940 kV | ABC | Warehouse2, PV2, Truck6 |

Topología: fuente bus1 → Tx → bus2; anillo **cerrado** bus2–bus3–bus4–bus5–bus6–bus2. No maniobras de interruptores ni operación isla (E:2001–2085).

| Línea | Extremos | Longitud | R/X por fase |
| --- | --- | ---: | --- |
| L23 | 2–3 | 50 m | 0.08 / 0.04 Ω/km |
| L34 | 3–4 | 150 m | 0.08 / 0.04 Ω/km |
| L45 | 4–5 | 20 m | 0.08 / 0.04 Ω/km |
| L56 | 5–6 | 50 m | 0.08 / 0.04 Ω/km |
| L62 | 6–2 | 150 m | 0.08 / 0.04 Ω/km |

La conversión m→km se hace correctamente. R y X son matrices diagonales, iguales entre fases, sin impedancias mutuas ni modelo de neutro. **No toda la matriz eléctrica carece de acoplamiento:** el LineCode no especifica CMatrix y conserva capacitancias por defecto: diagonal 2.8 y fuera de diagonal −0.6 nF/km, medidas en S2. El argumento c1=0 no se incorpora al comando. Las ampacidades leídas son 400/600 A normal/emergencia por defecto, sin evidencia de dimensionamiento para estos conductores (D:197–254).

**Transformador:** 800 kVA, 15/0.4 kV, uk=6 %, R total=1 %, XHL=√35=5.91608 %, %R=0.5 por devanado; wye/wye, sin desfase. 800 kVA es placa; R y grupo vectorial necesitan trazabilidad física. Se dejan otras propiedades a defaults del solver (E:593–630, D:156–175).

**Discrepancia de neutros comprobada:** aunque Rneut=−1 y los comentarios afirman “neutrales no aterrizados”, los buses se pasan sin sufijos de neutro. OpenDSS devuelve NodeOrder=[1,2,3,0,1,2,3,0] y NodeRef=[1,2,3,0,4,5,6,0]; ambos neutros quedan conectados directamente a tierra, nodo 0. Rneut negativo no elimina esa conexión. Esto cambia la interpretación de secuencia cero y desbalance. Evidencias: E:611–620, 2051–2053; D:162–175; S1/S2 y [documentación primaria Transformer](https://dss-extensions.org/dss-format/Transformer.html).

**Cargas:** 9 elementos monofásicos: tres por edificio, con P y Q independientes ABC; los Zoe se suman a las fases A/B de bus3 y no aparecen como dos elementos adicionales. No hay cargas bifásicas. PF de oficina uniforme [0.80,0.995], bodegas [0.57,0.88], fijo por episodio; Q=P·tan(arccos PF). Repartos por edificio constantes en el episodio, positivos y con cada peso entre 0.20 y 0.40; un evento puede cambiarlos (E:1055–1090, 1386–1391, 1972–1999, 2097–2106).

**DER:** PV1/PV2/PV3 con máximos 74.25/75.58/7.75 kW e inversores 100/100/10 kVA; BESS 250 kWh y 180 kVA; dos trucks de 315 kWh y 150 kVA. Son seis Load trifásicos de potencia negativa al inyectar, con P/Q totales repartidos por el modelo entre fases. No se usan objetos PVSystem/Storage ni un modelo dinámico de inversor OpenDSS: los SOC y límites viven en Python (E:2174–2189; D:298–320).

Todos los Load usan Model=1, Vminpu=0.80, Vmaxpu=1.10. **Estos dos valores delimitan la característica P/Q–V del modelo; no son límites de operación segura.** Fuera de ella puede cambiar la potencia realizada.

OpenDSS proporciona fasores mediante Bus.Voltages y Bus.Nodes; E convierte a magnitud, ángulo y pu con la base L-N de cada bus. PCC P/Q por fase procede del terminal de Vsource, con signo invertido; pérdidas de Circuit.Losses se convierten W→kW. Se calcula:
\[
VUF_i=100\frac{|V_{2,i}|}{|V_{1,i}|},\quad
V_1=(V_a+aV_b+a^2V_c)/3,\quad V_2=(V_a+a^2V_b+aV_c)/3.
\]
No se infieren ángulos ideales. En S1, bus5 en t=0 devolvió [0.9909137113, 0.9912422740, 0.9920350759] pu: diferencias reales entre fases. El mayor VUF LV de las semanas probadas fue ≈0.245 %, por tanto el desbalance nominal observado es pequeño, no severo. Los diagnósticos conservan ABC; **ningún voltaje, PF ni VUF entra en obs** (E:55–99, 2304–2356, 1233–1265).

## 4. Espacio de observaciones

Índices Python, inicio incluido y fin excluido. Orden de cargas: [P_A,P_B,P_C,Q_A,Q_B,Q_C]. Fuentes: E:427–445, 1233–1315; T:146–165.

| Variable | Índices | Dimensión | Unidad física | Normalización | Fuente |
| --- | --- | ---: | --- | --- | --- |
| Tiempo sin/cos semanal | 0:2 | 2 | Adimensional | sin/cos(2πt/168); no ciclo diario separado | t |
| Carga bus3, oficina + Zoe | 2:8 | 6 | kW, kvar | P/[40.74,40.74,18.74]; Q/14.055 | L + Z |
| Warehouse1 ABC | 8:14 | 6 | kW, kvar | P/65.66; Q/[65.66 tan(arccos .57)] | L |
| Warehouse2 ABC | 14:20 | 6 | kW, kvar | P/76.12; Q/[76.12 tan(arccos .57)] | L |
| PV disponibles efectivos | 20:23 | 3 | kW | /[74.25,75.58,7.75] | PV + evento |
| SOC BESS | 23:24 | 1 | fracción de 250 kWh | Sin otra transformación | B |
| Forecast P oficina | 24:35 | 11 | kW | /18.74; no incluye forecast Zoe | L |
| Forecast P warehouse1 | 35:46 | 11 | kW | /65.66 | L |
| Forecast P warehouse2 | 46:57 | 11 | kW | /76.12 | L |
| Forecast PV1 | 57:68 | 11 | kW | /74.25 | PV |
| Forecast PV2 | 68:79 | 11 | kW | /75.58 | PV |
| Forecast PV3 | 79:90 | 11 | kW | /7.75 | PV |
| Truck5 | 90:95 | 5 | bandera, SOC, horas, horas, SOC | [conectado,SOC,hasta salida/168,hasta llegada/168,max(.8−SOC,0)/.7] | T |
| Truck6 | 95:100 | 5 | Igual anterior | Igual anterior | T |
| SOC Zoe A/B | 100:102 | 2 | fracción de 52 kWh por EV | Sin otra transformación | Z |
| Precio actual | 102:103 | 1 | centavos USD/kWh | /10.13 | P |
| Forecast precio | 103:114 | 11 | centavos USD/kWh | /10.13 | P |

\[
n_s=2+18+3+1+66+10+2+1+11=114.
\]

No contiene P/Q del PCC, estados de red, último control, pérdidas, límites de corriente, eventos futuros explícitos, ni target terminal BESS. Los estados de truck incluyen su programación futura de conexión, supuesta conocida. El tiempo semanal identifica la posición finita del episodio. Por perfiles/ruidos latentes, no hay demostración de que obs sea un estado Markov suficiente; un agente MLP es una política sobre observaciones, no garantía de observabilidad completa.

**Límites:** tiempo [-1,1]; demandas actuales [0,+∞); forecasts y precio (−∞,+∞); PV y estados DER [0,1]. Los forecasts gaussianos pueden ser negativos o superar el pico; no se recortan. La realización PV sí se limita al máximo físico. _get_obs comprueba finitud antes de convertir a float32 y pertenencia al Box después. S1/S2: todas las observaciones de las tres semanas fueron finitas y válidas; Gym checker pasa y advierte de límites infinitos, no de observaciones ilegales.

**Futuro y leakage:** forecasts para t+1…t+11 construidos a partir de la realización futura exacta más ruido, no de un predictor causal. Esto es información futura deliberadamente aproximada, no un desfase accidental del índice. Hay riesgo de presentar estos forecasts como operacionalmente obtenibles: son un oráculo ruidoso con errores congelados por hora objetivo, sin revisión por horizonte. Los extremos no alteran los forecasts nominales; info de reset sí revela programa de eventos, por lo que un controlador que lea info podría obtener anticipación no permitida al agente. La extensión final reutiliza las primeras 11 horas de la misma semana. Véanse L:90–103, 147–171, 215–226; PV:161–166, 312–320; P:121–127, 139–155. Diferenciar esto de la contaminación train/test (§9).

## 5. Espacio de acciones y protecciones implícitas

\[
a_t=(u_{PV1},q_{PV1},u_{PV2},q_{PV2},u_{PV3},q_{PV3},
p_B,q_B,p_{T5},q_{T5},p_{T6},q_{T6}),\quad n_a=12.
\]

Cada coordenada pertenece a [-1,1], Box float32. No hay controles independientes por fase.

| Acción | Índice / dimensión | Rango RL | Conversión física efectiva | Unidad / DER |
| --- | --- | --- | --- | --- |
| Utilización PV1/2/3 | 0,2,4 / 3×1 | [-1,1] | P=(a+1)P_disponible/2; curtailment=(1−a)/2 | kW / cada PV |
| Q PV1/2/3 | 1,3,5 / 3×1 | [-1,1] | Q=a√(S²−P²), S=100,100,10 | kvar / cada PV |
| P BESS | 6 / 1 | [-1,1] | P solicitado inyectado=180a; después límites energéticos y de inversor | kW / BESS |
| Q BESS | 7 / 1 | [-1,1] | clip(180a,±min(180,√(180²−P_real²))) | kvar / BESS |
| P trucks | 8,10 / 2×1 | [-1,1] | P solicitado inyectado=150a; SOC limita y desconexión fuerza cero | kW / T5,T6 |
| Q trucks | 9,11 / 2×1 | [-1,1] | clip(150a,±√(150²−P_real²)); desconectado=0 | kvar / T5,T6 |

Signo público uniforme: P/Q de DER positivos inyectan. BESS interno u>0 carga, por eso se usa −180a; truck interno P_grid>0 consume, se invierte tanto en decoder como al alimentar snapshot. El PCC positivo importa. El Q positivo inyectado del DER se manda como kvar negativo de Load. Coherente en las rutas de producción y en las pruebas (E:1721–1759, 1787–1823; B:117–198; T:181–225).

BESS:
\[
E_{t+1}=E_t+\Delta t(0.95P_{ch}-P_{dis}/0.95),\quad
20\le E\le245\;\mathrm{kWh}.
\]
P_ch≤min(180,(245−E)/(.95Δt)); P_dis≤min(180,.95(E−20)/Δt); finalmente E se satura. SOC=E/250, no E/245. A SOC=.5 y Δt=1, descarga máxima real=99.75 kW y carga máxima=126.316 kW; no se pueden realizar 180 kW durante toda esa primera hora.

Truck: E=315SOC; .1≤SOC≤1; η=.95; energía integrada durante 1 h; llegada U(.2,.4); objetivo .8 al salir. Se recorta P según energía disponible, luego Q por círculo de 150 kVA. No se impone cargar lo necesario para alcanzar .8.

**Safety layer implícita local: sí.** Incluye clip de cualquier acción finita fuera del Box, límites de carga/descarga y energía, saturación Q y anulación de camiones desconectados. No hay corrección de voltaje, PF, corriente, VUF ni transformador basada en flujo de potencia. El decoder tolera no solo redondeo pequeño: también acciones muy grandes finitas se saturan. NaN/Inf y forma incorrecta se rechazan. info guarda la acción normalizada recortada y P/Q realizado, pero no conserva siempre la acción cruda ni una etiqueta global de intervención.

Acción cero **reduce PV al 50 %**. Para PV íntegro con BESS/trucks P=Q=0, el vector es [1,0,1,0,1,0,0,0,0,0,0,0]. No debe usarse el vector cero como “sin curtailment”.

## 6. Recompensa completa

En cada hora, exclusivamente:
\[
c_t=\frac{price^{raw}_t}{100},\quad
v^{PF}_{t,\phi}=\max(0,0.90-PF_{t,\phi}),
\]
\[
C_t=P^{grid}_t c_t\Delta t,\qquad
F_t=\lambda_{PF}\Delta t\sum_{\phi\in\{A,B,C\}}
v^{PF}_{t,\phi}|Q^{PCC}_{t,\phi}|,
\qquad \boxed{r_t=-C_t-F_t}.
\]

PF=|P|/hypot(P,Q), evaluado por fase del PCC; si la potencia aparente es ≤10⁻⁹, PF=1. El PF total también se informa, pero no sustituye los tres PF de la penalización. E:80–99, 147–217, 1835–1841.

| Término | Signo / coeficiente | Unidad | Escala observada por paso, seed 42 |
| --- | --- | --- | --- |
| Costo neto C_t | −1; precio raw/100 | USD/hora simulada | −0.858…14.044 (acción cero); −23.545…37.261 (aleatoria) |
| PF F_t | −lambda_pf; default 1 | USD/hora simulada | 0.328…29.759 (cero); 0…341.368 (aleatoria) |
| Voltaje | Ausente | — | 0 % directo |
| Pérdidas | Sin término separado; repercuten en P_grid | — | No separar un porcentaje aditivo sin contrafactual |
| SOC / déficit al salir / terminal | Ausente | — | 0 % directo |
| Acción, degradación, rampas, VUF, corriente, sobrecarga | Ausentes | — | 0 % directo |
| Curtailment | Sin penalización separada | — | Efecto económico a través del PCC |

S2, 168 h, lambda_pf=1:

| Política de prueba | Costo neto USD | PF USD | Return | Fracción PF de magnitudes absolutas |
| --- | ---: | ---: | ---: | ---: |
| Acción cero (PV al 50 %) | 693.558 | 937.364 | −1630.922 | 57.37 % |
| Uniforme, semilla acciones 4242 | 808.665 | 8709.922 | −9518.587 | 88.64 % |

La fracción se define como ΣF/(Σ|C|+ΣF), robusta cuando hay exportación. Si se divide por la suma neta de costos, se obtiene aproximadamente 57.47 % y 91.50 %, respectivamente; no confundir denominadores. Los primeros tres pasos nocturnos tuvieron ≈81–82 % PF. Son dos trayectorias ilustrativas, no una estadística de agentes entrenados.

La tarifa base varía 0.95–10.13 centavos USD/kWh, repetida diariamente y perturbada. La exportación recibe la misma tarifa que la importación. No hay compra/venta separada, peaje, demanda máxima ni costos de degradación (P:16–42, 103–127; E:195–201).

**No se encontró un SAC existente. Cualquier SAC conectado al ambiente por defecto recibiría penalizaciones eléctricas PF y no sería “unconstrained” respecto a esa métrica.** lambda_pf=0 elimina ese término sin modificar fuentes, pero conserva los límites físicos locales. El λ actual es un peso fijo; no es un multiplicador dual aprendido.

## 7. Definición efectiva de seguridad

| Restricción/condición | Evidencia | Dura / penalización / métrica | Termina episodio |
| --- | --- | --- | --- |
| Acción finita 12D | E:1677–1686 | Rechazo de inválidas; clip [-1,1] | Excepción, no flag |
| SOC BESS [.08,.98] | B:134–174 | Dura por limitación P y saturación E | No |
| SOC trucks [.10,1] | T:181–225 | Dura | No |
| Disponibilidad trucks | T:209–216 | P=Q=0 desconectado | No |
| P nominal DER | E decoder, B y T | Dura con defaults | No |
| P²+Q²≤S² | E:1728, B:149–166, T:220–224 | Dura con parámetros actuales | No |
| 0≤P_PV≤P_disponible | E:1724–1727; PV:479–484 | Dura; validada al registrar | No |
| SOC truck≥.80 al salir | T:278–295, E:1869–1870 | Brecha calculada, descartada por ambiente | No |
| SOC BESS final / balance energético de semana | E:1381, 1872–1943 | No implementado | Fin solo por tiempo |
| PF por fase≥.90 | E:197–200 | Penalización fija; brecha en info | No |
| Vmin≤V_iφ≤Vmax de seguridad | E:2344–2350 | No hay umbrales productivos; tensiones/extremos como métricas | No |
| Corriente máxima | D:245–252 | Sin comprobación; ratings DSS heredados | No |
| Transformador≤800 kVA | E:2373–2375 | Loading agregado como métrica | No |
| VUF máximo | E:2340–2356 | Solo porcentaje; no umbral | No |
| Convergencia | E:1825–1827 | Requerida para devolver transición | Excepción |
| Residual de balance pequeño | E:2384–2399 | Se calcula, no se exige tolerancia | No |

No existe función central is_safe ni cost vector para constrained RL. **Acción físicamente admisible al DER no equivale a acción segura para la red.** Actualmente “violación” explícita en reward significa brecha PF positiva, cuya severidad es (.90−PF)_+ y cuyo costo se pondera por |Q|. La misma brecha PF puede tener penalizaciones diferentes con distinto Q. Para voltaje/VUF/corriente no se define violación de producción.

Los umbrales .95/1.05 aparecen en diagnósticos previos, junto con otros umbrales exploratorios; no constituyen por sí solos un contrato del ambiente. En S2 se usan únicamente para contar: la acción aleatoria produjo 2/168 pasos fuera de ese intervalo, sin terminar el episodio.

## 8. Aleatoriedad y escenarios

| Componente | Qué cambia entre resets | Evidencia |
| --- | --- | --- |
| Oficina / bodegas | Selección uniforme de semana válida, reescalado por máximo de ESA semana, ruido real gaussiano σ=.10·base, mínimo cero | L:173–226 |
| Forecast cargas | Realización futura + ruido independiente σ=.10·|real|; una vez por reset | L:90–103 |
| PF y reparto ABC | PF uniforme por edificio; pesos por rechazo dentro [.20,.40], constantes en semana | E:1386–1391, 1949–1999 |
| PV | Una semana común; meteorología real=base+ruido .10; perturbación local AR(1), ρ=.95, σ estacionaria=.02, clip ±.05; cap físico por planta | E:1356–1380; PV:123–166, 284–320 |
| Precio | Mismo patrón 24 h, realizaciones y forecasts gaussianos .10 | P:103–127 |
| BESS | SOC inicial fijo .50 | E:1381 |
| Trucks | Calendario fijo; SOC nuevo U(.20,.40) en reset/llegadas | E:728–759; T:128–144, 283–284 |
| Zoe | 2 lun–vie, 1 sábado, 0 domingo; 9≤hora<18; SOC llegada U(.2,.4), carga hasta .8 a ≤22 kW | Z:18–29, 73–141 |
| Eventos extremos | None por defecto; un evento por semana, seleccionado al reset | E:967–1002; X:405–497 |

Eventos: surge 2–6 h ×[1.35,1.70] a un edificio; phase_imbalance 2–4 h con .55–.70 del total en una fase; collapse 2–6 h con PV multiplicado por [0,.30] en las tres plantas; compound 2–5 h con carga ×[1.35,1.60] y PV ×[0,.25]. “random” elige uniformemente uno de los cuatro, sin episodio nominal dentro de ese modo. Collapse/compound requieren una ventana solar diurna válida. Los cambios no afectan Zoe ni los forecasts (X:252–314, 427–496; E:1060–1111).

**Semillas:** super().reset(seed=seed) controla np_random. Nueve semillas de componente se extraen de él; precio, meteorología y eventos usan SeedSequence con dominios 10/11/12. Los samplers usan Generator propios y substreams. No hay random de Python ni RNG de PyTorch/SB3 en la transición. D utiliza uuid4 para nombres de LineCode: los nombres no se reproducen, pero la trayectoria eléctrica sí lo hizo.

**Resultado reproducibilidad:**
- reset(seed=42) + mismas acciones reprodujo exactamente observaciones, reward, flags e info de una trayectoria completa de 168 pasos: hash igual en S2.
- Semilla de action_space se debe fijar aparte para usar sample; los resets no la fijan. En la auditoría se generó una matriz de acciones con default_rng(4242).
- Constructor(seed=42).reset() no basta: reset sustituye el flujo de RNG por el de Gym, no inicializado con aquella semilla. Comprobado en S4.
- En modo nominal, dos instancias seedadas igual y luego reset(None) mantuvieron la misma observación.
- **En eventos aleatorios, reset(None) obtiene nueva entropía del sistema (E:983–990).** S4 dio programas distintos tras idéntico reset inicial: semillas de evento 1312053136 y 4174838624. Rompe la reproducibilidad de una ejecución con auto-reset al estilo entrenamiento, aunque cada episodio con seed explícita sí es reproducible.
- No se probó igualdad entre sistemas operativos, versiones del solver o hardware; se debe congelar el stack y usar tolerancias documentadas.

## 9. Training, validation y test

**No existe separación explícita; es un riesgo CRITICAL para un benchmark, no evidencia de que ya haya un entrenamiento contaminado.** reset muestrea todos los pools, no recibe conjuntos permitidos y options se ignora. Samplers permiten week_idx directo, pero el ambiente no expone split/scenario_id. Semillas distintas no evitan reutilizar el mismo perfil (E:1318–1380; L:184–195; PV:246–257).

S1 verificó:

| Pool | Filas originales | Filas usadas | Calidad / metadatos |
| --- | ---: | ---: | --- |
| office | 84×168 | 81 | 11 NaN en filas 38,41,56; se excluyen esas 3 semanas |
| warehouse1 | 70×168 | 70 | Finitas, sin metadatos de fechas |
| warehouse2 | 97×168 | 97 | Finitas, sin metadatos de fechas |
| solar | 80×168 | 80 | Finitas; solo metadata pv_floor |

No hay filas idénticas exactas dentro de cada pool según hash binario; esto no prueba ausencia de ventanas solapadas o similitud de perfiles. Los cuatro pools de CODIGO son idénticos byte a byte a sus copias canónicas en DATA_DIR, pero **el ambiente usa únicamente las copias bajo DATOS NATURE** (E:21–52).

Procedencia comprobada en S6:
- `extracciondatos_edificio_oficinas.ipynb` (historical reference; not distributed), celdas 2–8, selecciona Panther_office_Karla, agrupa semanas ISO completas y guarda weeks[20:]. Reconstrucción en memoria coincide exactamente con el NPZ, incluyendo NaN; primera semana retenida ISO 2016-21.
- `exploracion warehouses.ipynb` (historical reference; not distributed), celdas 2,5,6: Hog_warehouse_Louise y Rat_warehouse_Shari; bloques consecutivos de 168 filas; descarta >15 % ceros o suma<20. La reconstrucción coincide exactamente con ambos pools.
- El CSV empieza el **viernes 2016-01-01 00:00** y termina 2017-12-31 23:00, 17544 horas regulares. Por tanto las bodegas empiezan cada bloque en viernes, pero los vehículos interpretan t=0 como lunes; la oficina empieza en lunes. Desalineación comprobada, no conjetura.
- Existe solar_cleaned.csv, pero no se localizó un procedimiento de extracción que establezca qué serie/fechas producen el pool solar. No se atribuye una procedencia más precisa sin evidencia.

Propuesta posterior: recuperar fechas/identificadores originales, alinear calendarios y formar split cronológico por semana base, p.ej. 60/20/20, agrupando todas las perturbaciones de una semana en el mismo split. Aplicar embargo entre bloques cuando se reconstruyan ventanas/forecasts, deduplicar y preservar metadatos; reservar al menos una semana de separación si hay ventanas que crucen bordes. Congelar manifiestos y hashes; escenarios de validation/test deben utilizar exclusivamente semanas permitidas de TODOS los pools. Escalar con ratings fijos o estadísticas de training; documentar que el actual reescalado por pico semanal usa la semana completa y suprime variación estacional de amplitud. Nada de esto se implementó aquí.

## 10. Power flow, robustez y pruebas ejecutadas

Hay **un Solve explícito por step**, y además **CalcVoltageBases realiza una solución sin cargas** al reconstruir la red. Por semana son 168 soluciones operativas + 168 auxiliares de bases = 336 resoluciones contando ambas categorías. reset no llama al solver. S2 contó exactamente 504 Solve y 504 CalcVoltageBases en 504 transiciones; no hay búsqueda de acciones ni varios candidatos por paso. E:2084, 2192; D:85–98, 179–185; [semántica oficial de VoltageBases](https://dss-extensions.org/dss-format/Options.html).

D:37–64 activa excepciones y revisa el código de error después de cada comando. Fija LoadModel=PowerFlow, MaxIterations=500 y Tol=1e−9. El snapshot consulta Solution.Converged; si es False devuelve diagnósticos sin magnitudes eléctricas; step levanta RuntimeError antes del commit. No se omiten muestras silenciosamente ni se asigna un reward artificial. Sin un protocolo de entrenamiento para esa excepción, un fallo detendría la ejecución.

Se validan acciones, potencias de entrada, fasores, observación y datos de reward finitos; no existe una validación global de todo info ni una tolerancia obligatoria de balance. La convergencia no prueba seguridad ni igualdad entre potencias solicitadas y realizadas fuera del dominio de los Load.

| Verificación | Alcance y resultado |
| --- | --- |
| S1: reset + 3 pasos | Tres repeticiones, misma instancia y otra nueva, seed=42, acciones cero. 9 pasos convergentes, obs finitas/in-Box; igualdad exacta |
| S2: episodio y benchmark | 3×168 pasos: cero; acciones U[-1,1] con default_rng(4242); repetición de esa matriz. Todos convergentes y válidos; flags finales correctos; step posterior rechazado |
| S3: regresión breve existente | 43 tests, 4.351 s, todos OK: action_resolution, environment_step, electrical_instrumentation, reward, environment_reset_2c |
| S4: API y semillas | Gym check_env pasa con skip_render_check/skip_close_check; advertencias por Box infinito y pandas/NumPy. Reproducción nominal y fallo de seed del constructor / eventos en auto-reset confirmados |
| S5: tres snapshots artificiales | Sin avanzar dinámica; carga balanceada solo en wh1 de 1000/2000/3000 kW y Q=P/2. Demuestra transición de Load fuera de Vminpu, detallada abajo |
| S6: datos | Hashes/calidad NPZ y reconstrucción en memoria de oficina/bodegas desde CSV; coincidencia exacta y desalineación semanal comprobadas |

S3 usa [test_environment_step.py](<tests/test_environment_step.py>): tests de no convergencia y excepción en líneas 141–177 simulan el fallo y comprueban que tiempo/SOC no se confirmen. **No se observó una no convergencia real en los smoke tests.** No se ejecutó toda la suite, render ni las campañas históricas largas.

Detalle inicial, seed=42, acción cero:

| t | P_grid kW | Costo USD | PF USD | Reward | Vmin pu global | Mayor VUF LV % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 74.755697 | .773004 | 3.284060 | −4.057064 | .990594893 | .0333642 |
| 1 | 64.564185 | .692553 | 2.875145 | −3.567699 | .991856946 | .0290161 |
| 2 | 62.604249 | .618240 | 2.798911 | −3.417151 | .991951894 | .0305650 |

En las semanas completas: máximo |residual|=1.64e−7 kW (cero) y 4.63e−7 kW (aleatoria); Vmin/Vmax LV=.982446/.998907 y .937531/1.040158. Los valores LV de esta tabla excluyen bus1, mientras info.vmin/vmax global incluye la fuente.

**Error silencioso de interpretación fuera del dominio nominal:** S5 obtuvo:

| P solicitada kW | Q solicitada kvar | Converge | Vmin pu | Residual respecto a setpoints kW |
| ---: | ---: | --- | ---: | ---: |
| 1000 | 500 | Sí | .863307 | −0.000001 |
| 2000 | 1000 | Sí | .736935 | −390.034 |
| 3000 | 1500 | Sí | .665907 | −1172.548 |

El solver reduce las potencias por su característica de tensión; el snapshot mantiene entradas nominales para calcular el residual y no lo rechaza. Esto sería especialmente serio si se confirmasen SOC con setpoints distintos de la potencia eléctrica real de un DER. Son casos artificiales **fuera de las cargas nominales**, no evidencia de que todos los escenarios actuales sufran ese problema. La transición de modelo bajo Vminpu está documentada por [EPRI](https://opendss.epri.com/Vminpu1.html). E:2384–2399; D:9–10, 264–320. Se debe fijar un dominio operativo y tratar expresamente los resultados fuera de él.

Reproducción mínima sin archivos auxiliares, desde la carpeta CODIGO y con el intérprete indicado abajo:

```python
import hashlib, json
import numpy as np
from env.Ambiente_industrial import MicrorredIndustrial

actions = np.random.default_rng(4242).uniform(-1, 1, (168, 12)).astype(np.float32)

def serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: serializable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serializable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x

def fingerprint():
    env = MicrorredIndustrial()
    obs, _ = env.reset(seed=42)
    digest = hashlib.sha256()
    for action in actions:
        transition = env.step(action)
        obs, reward, terminated, truncated, info = transition
        assert env.observation_space.contains(obs) and np.isfinite(obs).all()
        assert np.isfinite(reward) and info["electrical_converged"]
        digest.update(json.dumps(serializable(transition),
                                 sort_keys=True, allow_nan=False).encode())
    assert terminated and not truncated and np.all(obs == 0)
    env.close()
    return digest.hexdigest()

assert fingerprint() == fingerprint()
```

Hash S2 de la trayectoria aleatoria completa: `e30d5cb7743af9d96cbb8931948746d27cd483d293de872866689564283467a1`. Incluye todos los outputs de step serializados; no promete identidad binaria con otro stack.

**Stack que funciona en esta máquina:** `Python anmrl` (original local reference omitted), Python 3.11.14; numpy 1.26.4; pandas 1.5.3; gymnasium **1.2.3**; OpenDSSDirect.py 0.9.4; dss-python 0.15.7; dss-python-backend 0.14.5; matplotlib 3.10.8. DSS C-API revisión 87d85c2622c8281b92255335bc7c09b11191b21d, basada en OpenDSS SVN 3723. SB3 2.7.1 y torch 2.10.0 están instalados, pero no se entrenaron ni cargaron agentes. Optuna no está instalado en anmrl.

El Python por defecto, `Anaconda base` (original local reference omitted), es 3.13.5 y carece de OpenDSSDirect. El runtime Python 3.12 incluido con la aplicación encontró paquetes del user-site, pero falló al importar pandas por ausencia de pytz. No se arreglaron esos entornos. La conclusión “funciona” depende de seleccionar el stack anmrl; no existe instalación autosuficiente documentada.

## 11. SAC existente e hiperparámetros

No se encontró ningún constructor SAC, llamada learn, checkpoint ni evaluación de un SAC dentro del repositorio. La instalación de una librería no demuestra que haya agentes entrenados.

| Hiperparámetro o artefacto solicitado | Evidencia de entrenamiento existente |
| --- | --- |
| Implementación usada por un experimento | No encontrada; SB3 está disponible en anmrl |
| Política, actor/críticos, capas y neuronas | No definidos en el repositorio |
| Learning rate | No definido |
| Batch size / replay buffer | No definidos |
| Gamma / tau | No definidos |
| Entropy coefficient / target entropy | No definidos |
| Learning starts | No definido |
| Train frequency / gradient steps | No definidos |
| Total timesteps | No registrado |
| Normalización de entrenamiento | Solo normalización física del ambiente; sin VecNormalize |
| Callbacks / criterio de evaluación | No encontrados para RL |
| Seeds de entrenamiento / torch / SB3 | No hay protocolo |
| Checkpoint o curva de aprendizaje | No encontrados |
| Baseline razonablemente entrenado | **No** |

No se rellenan estas casillas con los defaults de Stable-Baselines3: serían parámetros de una librería, no evidencia de experimentos. La propuesta futura está en F.

## 12. Lagrangian SAC / constrained RL

**No hay implementación funcional encontrada.** No hay crítico de costos, actualización dual, PID, budgets de constraints, estado λ por episodio ni almacenamiento de costos en replay.

El único lambda_pf (E:528, 554–557, 1655–1661) es un coeficiente externo fijo usado en reward:
\[
\lambda^{actual}_{t+1}=\lambda^{actual}_t
\]
durante la ejecución normal. No mide error respecto a un presupuesto ni se actualiza por gradiente. K_P, K_I, K_D, λ_0 adaptativo y λ_max no existen.

El costo PF actual pondera por |Q|. Controlar su media **no equivale** a controlar la fracción de horas con PF<.9, tampoco el voltaje, el VUF o los incumplimientos SOC. Sería incorrecto llamar Lagrangian SAC a SAC con ese peso fijo.

Para la fase siguiente se propone un SAC con costos separados y multiplicadores actualizados; G especifica una posibilidad. PID es una alternativa adicional, no una implementación ya presente; el trabajo original de [Stooke, Achiam y Abbeel](https://proceedings.mlr.press/v119/stooke20a.html) distingue control integral y términos proporcionales/derivativos. No es necesario introducir PID para el primer contraste mínimo.

## 13. Optuna previo

No se localizaron importaciones Optuna, create_study, study.optimize, bases de estudios, trial logs ni best_params. Optuna tampoco está instalado en el intérprete usado para los smoke tests.

| Elemento | Situación actual |
| --- | --- |
| Sampler / pruner | No encontrados |
| Número de trials | Sin estudio; no hay número histórico verificable |
| Objetivo / search space | No definidos |
| Timesteps por trial / evaluación | No definidos |
| Seeds / almacenamiento | No definidos |
| Mejores hiperparámetros | No existen resultados encontrados |
| Reutilización para A/B | No hay protocolo reutilizable; sí se puede reutilizar el ambiente después de resolver bloqueos |

No se ejecutaron ni crearon estudios. El protocolo de F/G es una propuesta nueva, con budgets iguales para A y B, sin modificar umbrales de seguridad durante el tuning.

## 14. Métricas disponibles

**Disponible en info no significa registrado en disco.** E:1880–1937 entrega telemetría; el caller debe agregarla y persistirla. render_mode=None no guarda historia. El dashboard opcional almacena solo un subconjunto (E:1453–1493).

| Métrica | Estado | Cómo obtenerla / limitación |
| --- | --- | --- |
| Episodic return | Derivable | Σinfo.reward; no logger episódico RL |
| Costo económico | Disponible por paso | operating_cost_usd; agregar semana |
| Penalización PF | Disponible | pf_penalty_usd, pf_violation_phase |
| Energía importada/exportada | Derivable | Σmax(±p_grid_kw,0)·dt; distinguir importación de energía neta |
| Pérdidas | Disponible por paso | p_losses_kw, q_losses_kvar; energía activa=ΣP_loss·dt |
| Renewable utilization | Derivable | ΣPV realizada / ΣPV disponible efectiva; tratar denominador cero |
| Curtailment por control | Derivable | Σ(P_disponible_efectiva−P_inyectada)·dt |
| Reducción PV por evento | Derivable | Diferencia nominal−efectiva; NO mezclarla con curtailment voluntario |
| Tensiones | Disponible ABC | voltages_pu, voltage_angles_deg; Vmin/Vmax y bus/fase |
| Voltage violations | Por implementar contrato y agregación | Fijar umbrales; contar horas y pares bus-fase por separado |
| Severidad voltaje | Derivable tras fijar umbrales | max(0,Vmin−V,V−Vmax); máximos y sumas pu·h |
| PF violations | Disponible/derivable | brecha por fase, conteos y duración |
| SOC físico | Disponible parcialmente | BESS/trucks before/after, Zoe SOC; contabilizar con disponibilidad |
| SOC departure violations | **Falta integrar** | Retorno de advance_interval descartado; no métrica de salida en info |
| SOC terminal BESS | Derivable desde controls final | Sentinel no contiene SOC físico; falta objetivo terminal |
| VUF | Disponible | Por bus; max global y LV; falta umbral |
| Sobrecarga Tx | Disponible agregado | transformer_loading_pct; no corriente máxima por fase |
| Corrientes de líneas / neutro | No en info | Añadir extracción ABC; no hay neutro explícito de red |
| Convergencia | Disponible si retorna | electrical_converged; en fallo step lanza excepción y no retorna info |
| Error de balance | Disponible | balance_residual_kw; no se hace cumplir una tolerancia |
| Acción aplicada / protecciones | Parcial | action_clipped y controls; faltan acción cruda, causas y magnitud de intervención |
| Runtime | Solo scripts diagnósticos / S2 | Falta logger de wall time por entrenamiento, evaluación y step |

Claves adicionales de info: t, next_t, terminated, truncated, terminal_observation_is_sentinel, price_raw, price_usd_per_kwh, pcc_p_kw/q_kvar/pf, pcc_p_kw_phase/q_kvar_phase/pf_phase y voltage_phase_order. controls contiene los seis DER y Zoe; loads separa office, zoe, bus3, wh1, wh2 y total, cada uno con P/Q ABC y total. **No sumar bus3 además de office+zoe**, ni “total” a sus componentes. Metadata de eventos incluye enabled, active, type, start/end, target, intensity y phase. E:1004–1044, 1157–1202, 340–407.

El snapshot privado conserva también fasores complejos, element_powers, balances P/Q y potencias/pérdidas de ambos lados del transformador; parte de ello no pasa a info. No usar estos accesos privados para dar información adicional al agente sin modificar formalmente su contrato.

Resultados históricos relevantes:
- `transformer_episode_diagnostics_vminpu_080_results.json` (historical reference; not distributed): 20 episodios, 3360 horas aleatorias, 0 no convergencias, 70 horas con Vmin<.95; mayor residual de setpoints ≈7.05e−7 kW. La configuración declara que no llamó env.step.
- `transformer_episode_diagnostics_results.json` (historical reference; not distributed): versión previa del modelo; llegó a discrepancias de potencia ≈13.89 kW. No mezclar como si fuera el mismo ambiente.
- `stress_voltage_analysis_results.json` (historical reference; not distributed): 20000 snapshots instantáneos, escenarios LV y sensibilidad del transformador; su metadata distingue factibilidad instantánea de factibilidad SOC intertemporal.
- `vminpu_experiment_results.json` (historical reference; not distributed) y `vminpu_experiment_before.json` (historical reference; not distributed): exploración de la característica de cargas.
Estos archivos fueron leídos, no regenerados, y no demuestran rendimiento de un SAC.

## 15. Auditoría de corrección: hallazgos priorizados

CRITICAL significa que invalida el benchmark propuesto o su objetivo económico/operacional; no necesariamente que Python deje de ejecutar.

| ID | Nivel | Hallazgo y consecuencia | Evidencia |
| --- | --- | --- | --- |
| C01 | CRITICAL | No hay separación de perfiles train/val/test. Un split por semillas puede reutilizar semanas vistas e invalidar generalización | §9; E:1346–1380 |
| C02 | CRITICAL | La brecha SOC de salida de trucks se descarta. El agente no tiene obligación efectiva de prestar ese servicio: puede explotar energía de llegadas sin reponerla antes de salir | T:278–295; E:1869–1870, reward 195–201; S2: 8 salidas deficitarias en ambas políticas |
| H01 | HIGH | Neutros realmente a tierra, pese a descripción flotante. La representación de secuencia cero publicada sería incorrecta | D:162–175; E:611–620; NodeRef/NodeOrder S1/S2 |
| H02 | HIGH | Bodegas con semanas viernes→jueves y vehículos lunes→domingo; calendario industrial incoherente | Notebooks de extracción y reconstrucción exacta S6 |
| H03 | HIGH | Etiquetar SAC por defecto como unconstrained sería incorrecto por penalización PF fija | E:195–201; S2 |
| H04 | HIGH | No contrato productivo de seguridad ni costos de constraints V/I/VUF/Tx. No se puede afirmar “seguro” solo por convergencia o clipping de DER | §7 |
| H05 | HIGH | Eventos con reset(None) usan entropía nueva: una semilla inicial de entrenamiento no reproduce todos los episodios extremos | E:983–990; S4 |
| H06 | HIGH | Sin objetivo/valor terminal BESS; cada semana reinicia 125 kWh sin contabilidad terminal. Puede favorecer liquidar energía al final | E:1381, 1872–1943; B:8–18 |
| H07 | HIGH | Forecasts basados en futuros realizados y normalización al pico de cada semana; no equivalen a predicción causal ni a distribución histórica de amplitudes | L:200–223; PV:116–120, 284–317; P:117–127 |
| H08 | HIGH | No instalación fijada ni empaquetado; los intérpretes locales producen fallos distintos. Entrenamiento/reproducción dependen del entorno correcto | §10; ausencia de requirements/pyproject del ambiente |
| M01 | MEDIUM | Snapshot convergente puede tener gran discrepancia de setpoints fuera de [.80,1.10] pu; no hay guard de residual ni reconciliación energética | S5, E:2384–2399; crítico si ese dominio se admite en futuros escenarios |
| M02 | MEDIUM | Capacitancia y ampacidades de líneas heredadas; c1=0 no elimina CMatrix. Falta trazabilidad térmica | D:211–252; S2 |
| M03 | MEDIUM | Constructor seed no controla reset sin argumento, a pesar de exponer ese parámetro | E:572–573,1320–1325; S4 |
| M04 | MEDIUM | Un contexto global OpenDSS y Clear por snapshot. Hilos concurrentes compartirían y borrarían red activa; no hay aislamiento por env | D:2,37–41; evitar concurrencia en hilos sin validación |
| M05 | MEDIUM | PCC agregado y loading Tx total pueden esconder estrés de una fase; falta corriente y carga por devanado/fase | E:2373–2375; §14 |
| M06 | MEDIUM | Tarifa de origen externo sin referencia concreta, compra=venta y sin costo de degradación; limita interpretación económica | P:1–6,16–42; E:195–201 |
| M07 | MEDIUM | Datos sin IDs/fechas persistentes y extracción solar no trazada; licencia del ambiente no declarada | §9, §17 |
| M08 | MEDIUM | info anticipa evento completo desde reset; comparar controladores con accesos distintos produciría ventaja informativa | E:1004–1044,1419 |
| M09 | MEDIUM | Un error del solver aborta en vez de seguir un contrato de evaluación; descartarlo posteriormente sesgaría seguridad | E:1825–1827 |
| L01 | LOW | Comentarios de “decoder futuro”, “step futuro” y clase incompleta están obsoletos | E:414–420,475,761–762 |
| L02 | LOW | Parámetros week_noise_sigma/phi y clip_nonnegative del sampler de carga no gobiernan la nueva ruta como sugieren sus nombres; el clipping real a cero es incondicional | L:24–26,45–47,200–223 |
| L03 | LOW | Notebook prueba camion bus5 importa V2GTruckBus5, símbolo inexistente actualmente | notebook celda 0; T:26 |
| L04 | LOW | APIs legacy de forecast incluyen t, mientras forecast_at empieza en t+1; riesgo al reutilizar ejemplos | L:147–171 vs 233–247; no afecta step actual |
| L05 | LOW | allow_q_when_disconnected=True no aplica Q: solo cambia q_max reportado; producción usa False | T:209–216 |
| L06 | LOW | Parámetros físicos asimétricos futuros requerirían revisar escalado: BESS usa P_dis_max para ambos signos, truck p_ch_max para ambos | E:1738; T:220; defaults actuales son simétricos |

C02 es un defecto de integración confirmado. S2: Truck5 sale a t=7 con SOC .2423 bajo acción cero, déficit .5577; ese valor no aparece en reward ni info. La política aleatoria también tuvo 8 salidas deficientes. No confundir el target .8 con un SOC mínimo duro de .1: se cumple el segundo y se incumple el servicio.

Verificaciones positivas: kW/kvar y pérdidas W→kW correctos; bases L-L/L-N correctas por bus; signos DER/PCC coherentes; SOC BESS respecto a capacidad nominal; energía integrada una sola vez; precio y potencias del mismo t; observación siguiente en t+1; forecasts no avanzan RNG al consultarlos; terminal semanal explícito y bloqueo de step extra. Evidencia: D:257–262,405–430,475–477,529–546; E:1762–1947; S1–S3. No se encontraron bugs de esos tipos en las rutas y parámetros nominales probados.

## 16. Rendimiento computacional

Equipo: Intel Core i5-12500H, 12 núcleos / 16 procesadores lógicos; un proceso Python, sin entrenamiento ni render. S2 instrumentó comandos y midió cada step; las mediciones incluyen una pequeña sobrecarga del contador. Las semanas completas además serializaron outputs para el hash, fuera del cronómetro de step.

| Trayectoria | Media step | Mediana | p95 | Tiempo total 168 pasos |
| --- | ---: | ---: | ---: | ---: |
| Cero | 16.95 ms | 15.61 ms | 26.11 ms | 2.910 s |
| Aleatoria | 19.39 ms | 19.86 ms | 25.21 ms | 3.333 s |
| Aleatoria repetida | 18.81 ms | 19.74 ms | 25.08 ms | 3.231 s |

Constructor inicial ≈16.9 ms; reset medido ≈1.7–6.2 ms. Hay variación de carga del equipo; presupuesto prudente del **ambiente solo**: 17–20 ms/step, 50–59 steps/s, ≈2.9–3.4 s/semana. No es un benchmark de entrenamiento SAC.

504 transiciones generaron 20664 comandos: 41 por paso (1 Clear, 27 New, 6 Set, 1 Edit, 4 continuaciones, 1 CalcVoltageBases, 1 Solve). Solo Clear consumió 2.457 s acumulados; Solve 0.074 s y CalcVoltageBases 0.080 s. El resto incluye llamadas API y procesamiento Python.

Perfil cProfile de 12 pasos: 0.403 s, de los que 0.379 s (94 %) están en _solve_electrical_snapshot; construcción 0.079 s; extracción PCC dos veces por paso 0.094 s; get_power_balance 0.079 s; muchas construcciones de DataFrame y Series. **Estos tiempos son acumulativos y se solapan: no sumarlos.** El cuello dominante es reconstrucción/limpieza más extracción/tabulación de diagnósticos, no la iteración numérica pura del Solve. D:39–51,451–492,495–553; E:2146,2259,2358.

No se optimizó nada. Antes de paralelizar, procesos/contextos separados por solver; no asumir velocidad lineal ni usar hilos compartiendo el contexto global. La estimación completa de campañas, con hipótesis explícitas, está en H.

## 17. Preparación para liberar un benchmark publicable

| Requisito | Estado | Justificación / trabajo pendiente |
| --- | --- | --- |
| API reset/step y espacios | READY | Contrato Gymnasium comprobado, 114/12, flags y sentinel |
| Registro Gym y configuración serializable | MINOR WORK | Actualmente instanciación directa y parámetros internos fijos |
| Modelo explícito ABC y métricas eléctricas | READY | Fasores reales, PCC por fase, VUF y pérdidas verificados |
| Fidelidad topológica / neutro / parámetros | MAJOR WORK | Resolver H01, justificar modelo sin neutro y parámetros heredados |
| Semillas nominales explícitas | READY | Trayectoria completa reproducida en mismo stack |
| Semillas de campaña y eventos | MAJOR WORK | Corregir auto-reset extremo y unificar contrato |
| Documentación del ambiente | MAJOR WORK | No hay README de instalación/uso propio; comentarios obsoletos |
| Ejemplo mínimo actualizado | MINOR WORK | El código de §10 es reproducible; reparar notebooks legacy |
| Instalación / requirements / lock | MAJOR WORK | Falta paquete propio, versiones, CI y prueba en entorno limpio |
| Datos y trazabilidad | MAJOR WORK | Fechas, selección solar, sincronía de semanas y rutas canónicas |
| Licencias / atribución | MAJOR WORK | Falta licencia explícita para este ambiente y manifiesto de datos |
| Split train/validation/test | MAJOR WORK | No existe |
| Agentes baseline | MAJOR WORK | No hay SAC/Lagrangiano ni política de servicio completa |
| Métricas/evaluación persistente | MAJOR WORK | info útil, pero sin protocolo ni resultados RL |
| Pruebas existentes | READY | Base unittest sólida; 43 seleccionadas pasan; falta integrar CI de release |
| Dependencia OpenDSS | MINOR WORK | Backend identificado y funciona; fijar versión, plataforma e instrucciones |
| Pregunta científica / protocolo | MAJOR WORK | Fijar seguridad, servicio y generalización antes de afirmar superioridad |

El `LICENSE del repositorio BDG2 incluido` (historical reference; not distributed) contiene MIT y atribución a BUDS Group; es un archivo del componente de datos/software upstream. No se encontró licencia propia en la raíz/CODIGO y no se debe inferir que aquella cubra automáticamente la autoría nueva, todos los datos derivados o una tabla eléctrica de un paper no identificado. Preparar atribuciones y términos aplicables por componente.

## 18. Diseño preliminar de experimentos — propuesta, no ejecutada

Pregunta defendible: **¿qué costo económico y qué generalización a semanas no vistas ofrece SAC frente a SAC con costos eléctricos explícitos, bajo el mismo contrato de servicio y las mismas protecciones locales de DER?** Separar resultados nominales de eventos extremos, sin cambiar información del agente.

**Condiciones previas comunes:** resolver C01/C02 y los HIGH que alteren escenarios/física; fijar versión del ambiente; servicio V2G y condición energética terminal comunes; umbrales y budgets de seguridad congelados; mismas observaciones, action mapping, escenarios y potencia disponible. La ausencia de violaciones de SOC después del clipping no demuestra que RL haya aprendido seguridad de SOC.

- **Baseline 0:** existe operación pasiva diagnóstica: PV íntegro, Q=0, BESS/trucks sin P/Q, en `transformer_episode_diagnostics.py` (historical reference; not distributed). Es referencia eléctrica, pero incumple el servicio V2G. Para comparación económica justa, construir una regla sencilla que cargue trucks para el target de salida, use PV íntegro y una regla BESS predefinida compatible con el cierre energético. No posee aprendizaje ni tuning con test.
- **Baseline 1:** SAC con reward económico común y **sin penalizaciones eléctricas**; lambda_pf=0. Se conservan los límites físicos de DER. El contrato de servicio debe estar definido antes para que no gane simplemente por no cargar vehículos.
- **Baseline 2:** SAC con el mismo reward económico, costos eléctricos separados, críticos de costo y actualización dual. Evitar conservar también la penalización fija PF dentro de reward, pues duplicaría el objetivo.

Como contrato inicial propuesto, sujeto a aprobación científica posterior: V∈[.95,1.05] pu para los 15 pares bus-fase LV; PF≥.90 en las 3 fases PCC. No se presentan estos números como cumplimiento normativo. Mantener VUF, loading y corrientes como resultados explícitos; no afirmar seguridad sobre ellos si no se convierten en constraints con límites defendidos.

**Tuning (selección de hiperparámetros):** comenzar con 20 trials por algoritmo; ampliar simétricamente a 30 si el presupuesto y la inestabilidad lo justifican. 100800 pasos por trial = 600 semanas simuladas; evaluación cada 16800 pasos en 20 escenarios fijos de validation. Son seis evaluaciones, sin episodios de test. Una seed de aprendizaje común para el barrido inicial; revisar las mejores configuraciones con validación adicional predefinida si hay alta variabilidad, contabilizando ese costo. Las tablas F/G restringen el número de dimensiones ajustadas.

Para A seleccionar menor costo económico medio de validation; reportar seguridad sin usarla como filtro oculto de ese SAC “libre”. Para B seleccionar menor costo entre configuraciones que satisfacen budgets de validation; si ninguna cumple, reportar “no factible”, no relajar los budgets retrospectivamente. Para pruning se puede usar orden factibilidad→exceso normalizado→costo, con escala/regla predefinida y conservando los componentes originales.

**Confirmation (reentrenar, no seguir tuneando):** congelar una configuración por algoritmo, entrenar desde cero con 5 seeds de aprendizaje, 504000 pasos por seed (=3000 semanas). Seeds de red, replay/exploración y secuencias de escenarios registradas por separado. No elegir únicamente la mejor seed. Usar mismo presupuesto y criterio de checkpoint; curvas de aprendizaje y dispersión por seed.

**Final evaluation:** congelar checkpoints; evaluar cada seed en 100 escenarios nominales de semanas base reservadas, emparejados entre métodos. Para estrés, 25 escenarios por tipo ×4=100 adicionales, también de pools de test, más un nominal emparejado para cada caso si se estudia efecto causal del evento. Los 100 escenarios no son 100 semanas históricas independientes: pueden reutilizar semanas de test con distintos ruidos. Informar ambos conteos y agrupar intervalos estadísticos por semana base y seed; no tratar 168 horas correlacionadas como 168 muestras independientes.

Métricas finales: costo USD, import/export kWh, return descompuesto, pérdidas kWh, utilización PV y curtailment voluntario, horas y pares bus-fase con violación, severidad máxima/acumulada y duración consecutiva, PF por fase, déficit energético/SOC al salir, energía terminal, VUF, sobrecarga/corriente, fallos de solver, intervención de límites locales y tiempo de cómputo. Evaluación con acciones deterministas de la política como protocolo principal, igual para ambos SAC; sensibilidad estocástica solo si se predefine.

Las magnitudes anteriores son presupuestos iniciales de trabajo, **no evidencia de convergencia del aprendizaje**. No se incluyen conformal prediction ni proyección de seguridad/QP en este diseño.

## 19. Objeciones de un revisor y mitigación

| Objeción | Juicio de auditoría | Mitigación |
| --- | --- | --- |
| “No es trifásico” | Refutable: 18 nodos de fase, fasores y potencias ABC reales | Publicar snapshot, topology y prueba ABC |
| “No es desbalanceado / es demasiado leve” | Desbalance real pero VUF nominal pequeño en S2 | Reportar distribución por fase; estrés de reparto separado, sin exagerar severidad |
| “El neutro está mal descrito” | Objeción fundada: descripción flotante contradice nodo 0 | Decidir modelo físico y publicar conexiones efectivas |
| “El SAC baseline está mal ajustado” | Aún no existe | Tuning comparable, múltiples seeds y curvas con budget explícito |
| “El test fue visto durante training” | Riesgo CRITICAL actual | Split por perfiles/fechas y hashes, no solo por seeds |
| “Los forecasts conocen el futuro” | Oráculo ruidoso explícito | Declarar supuesto; evaluar errores de forecast; no alegar predictor causal |
| “Las restricciones ya están en reward” | PF ya penalizado por defecto | Reward económico separado, eliminar doble penalización en comparación |
| “La seguridad se debe al clipping” | Cierto para límites de DER | Igual decoder en todos, registrar intervenciones y medir seguridad de red aparte |
| “La política gana por no cargar vehículos” | Riesgo confirmado C02 | Exigir servicio común y reportar déficit al salir |
| “Hay energía gratis al cerrar la semana” | No hay condición/valor terminal | Cierre energético o valor terminal justificado e igual en todos |
| “Los calendarios no coinciden” | Bodegas empiezan viernes | Reconstruir y alinear timestamps antes del split |
| “No existe baseline simple” | Solo referencia pasiva sin servicio | Regla determinista de energía/servicio, accesos informativos iguales |
| “Convergencia oculta shedding implícito” | Posible fuera de Vminpu, S5 | Verificar potencias reales/residual y definir dominio y fallo |
| “Las métricas son agregadas y esconden fases” | Algunas sí, p.ej. Tx | Publicar ABC y corriente de la fase más cargada |
| “Los resultados previos no evalúan env.step” | Correcto en campañas históricas | Separar validación eléctrica de evaluación RL end-to-end |
| “No es reproducible ni instalable” | Seed explícita sí, release aún no | Lock, instalación limpia, CI, manifiestos y corrección de eventos |
| “Solo es software sin pregunta científica” | Riesgo si se publica así | Contrastar costo-seguridad y generalización con hipótesis y análisis estadístico |

## 20. Cierre y plan posterior

Las secciones A–J siguientes son el cierre operativo de esta auditoría. No autorizan implementación ni entrenamiento.

## A. Qué tenemos hoy

1. Ambiente Gymnasium funcional de 114 observaciones y 12 acciones.
2. Episodios finitos de 168 horas con integración temporal coherente.
3. Seis buses, 18 nodos ABC y anillo LV con transformador.
4. Flujo trifásico desbalanceado real, con neutros efectivos a tierra.
5. Tres edificios, tres PV, un BESS, dos camiones V2G y dos Zoe.
6. P/Q y SOC con límites físicos locales anteriores al flujo.
7. Reward económico con penalización fija PF por fase.
8. Forecasts sintéticos de once horas derivados de futuras realizaciones.
9. Escenarios nominales y cuatro clases de eventos extremos.
10. Reproducción exacta de una semana con seed explícita en el stack probado.
11. Amplia telemetría eléctrica y dashboard opcional.
12. 43 pruebas seleccionadas aprobadas y resultados eléctricos históricos.
13. Ningún SAC entrenado, SAC Lagrangiano ni estudio Optuna encontrado.
14. Ningún split formal de datos.
15. Simulación de una semana en aproximadamente tres segundos, sin aprendizaje.

## B. Problemas que deben corregirse antes de entrenar

**CRITICAL**
- C01: separar perfiles train/validation/test con manifiesto trazable.
- C02: integrar el requisito de SOC/energía de salida V2G en el contrato operacional común.

**HIGH**
- H01/H02: decidir neutros y alinear calendarios; congelar la física y escenarios resultantes.
- H03/H04: separar reward económico y costos eléctricos, definir acción/violación segura y budgets.
- H05: asegurar reproducibilidad de campañas con eventos y auto-reset.
- H06: definir condición o valor energético terminal.
- H07/H08: declarar/corregir supuesto de forecast y fijar instalación reproducible.

**MEDIUM**
- M01–M09: dominio válido del solver/residuales, parámetros de línea, seed del constructor, aislamiento de contextos, métricas térmicas por fase, economía, procedencia/licencia, información de eventos y protocolo ante fallos.
- M01 debe elevarse a bloqueo de entrenamiento si se admiten escenarios fuera del dominio de potencia constante observado.

**LOW**
- L01–L06: actualizar comentarios y ejemplos; retirar/aclarar parámetros inactivos y APIs legacy; revisar opciones de Q desconectado y escalados si se generalizan parámetros.

No se detectó un bug CRITICAL de unidades o signo en la ruta nominal. Sí un defecto CRITICAL de integración del servicio V2G y un bloqueo metodológico CRITICAL del split; no son equivalentes y deben reportarse por separado.

## C. Qué NO debemos tocar

- Preservar el contrato ABC explícito y el cálculo de VUF a partir de fasores reales.
- Preservar bases de tensión por bus y conversiones L-L/L-N y W/kW correctas.
- Preservar signos públicos de DER/PCC y sus pruebas.
- Preservar resolve→snapshot→reward→commit, sin doble integración de SOC.
- Preservar consultas idempotentes, forecast_at(t) desde t+1 y alineación precio/potencia de la misma hora.
- Preservar validación de acciones/observaciones y sentinel terminal documentado.
- Preservar aislamiento de RNG nominal por componente y meteorología común entre PV.
- Preservar clipping físico y capability de DER de forma idéntica entre baselines; instrumentarlo.
- Preservar separación de cargas office/Zoe/bus3 en telemetría y pruebas existentes.
- Mantener intactos los datos y resultados originales para trazabilidad; cualquier regeneración futura debe versionarse.

Estas son partes verificadas, no una prohibición de refactorizar posteriormente si se conserva el comportamiento y se valida equivalencia.

## D. Experimentos que ya podemos ejecutar

- Smoke tests nominales, regresión y reproducibilidad por seed explícita.
- Diagnósticos eléctricos de signos, bases, balances, fasores y acción aplicada.
- Comparación diagnóstica de control cero, PV íntegro y acciones aleatorias, declarando que no cumplen necesariamente servicio.
- Análisis descriptivo de pools/fechas, intensidad de eventos y escalas de reward.
- Perfilado pequeño y medición de métricas derivadas.

Los tests de esta auditoría ya cubren una parte de ello. No equivalen a un benchmark publicable de RL.

## E. Experimentos que todavía NO debemos ejecutar

- Optuna masivo y entrenamientos finales de SAC/Lagrangian SAC.
- Comparación llamada “unconstrained vs constrained” con reward PF mezclado.
- Evaluación final con el mismo pool de perfiles de entrenamiento.
- Estudios económicos que ignoren salidas V2G y energía terminal.
- Campañas de eventos que confíen solo en una seed inicial con auto-reset actual.
- Entrenamientos paralelos en hilos sobre el OpenDSS global.
- Afirmaciones de seguridad de corriente/VUF/transformador sin contrato y métricas por fase.
- Comparación directa de resultados históricos LV/snapshots con retornos del step actual.

## F. Propuesta de Optuna para SAC

**Propuesta nueva; ningún trial ejecutado.** SAC SB3 MlpPolicy con λ_pf=0 para el componente eléctrico, después de fijar el objetivo de servicio común. Normalización física actual; sin aprender estadísticas con validation/test. Reward económico escalado opcionalmente por una constante fija común elegida en training, sin ajustar esa escala por trial.

| Hiperparámetro | Rango inicial propuesto | Tratamiento |
| --- | --- | --- |
| learning_rate | 1e−4 a 1e−3 | Log-uniforme |
| net_arch | [128,128] o [256,256] | Dos capas ocultas, ReLU; mismo tamaño actor/críticos |
| batch_size | 128 o 256 | Categórico |
| gamma | .99, .995 o .999 | Categórico; sensibilidad al horizonte semanal |
| tau | .005 | Fijo inicialmente; si se amplía: .002–.02 log |
| buffer_size | 200000 | Fijo; no llenar search space con buffer en solo 20 trials |
| learning_starts | 3360 o 8400 | 20/50 episodios; categórico |
| train_freq | 1 paso | Fijo |
| gradient_steps | 1 | Fijo; expansión 1/2 solo después de medir costo de actualización |
| ent_coef | auto_0.1 | Fijo inicialmente; expansión auto_0.01 / auto_0.1 / auto_1.0 |
| target_entropy | −12 | Fijo inicial, correspondiente a 12 acciones |
| target_update_interval | 1 | Fijo |
| n_steps / use_sde | 1 / False | Fijos |
| total_timesteps | 100800 | Budget por trial, no variable optimizada |
| seed de aprendizaje | Lista registrada separada del split | No elegir la seed ganadora como hiperparámetro |
| lambda_pf del ambiente | 0 | Fijo para A, no parámetro Optuna |

Semántica de parámetros contrastada con [SB3 SAC 2.7.1](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/sac.html). Los rangos son juicio de diseño inicial de esta auditoría, no valores óptimos publicados para este ambiente.

Sampler propuesto: TPESampler con seed fijada y 5 trials iniciales aleatorios. Pruner: MedianPruner, sin podar antes de 33600 pasos (dos checkpoints), n_startup_trials=5. Empezar serialmente para reproducir decisiones de búsqueda; guardar estudio, parámetros, versiones, split y métricas por trial. SQLite local puede servir a un proceso; reconsiderar almacenamiento si se decide concurrencia. Referencias de interfaces: [TPE](https://optuna.readthedocs.io/en/latest/reference/samplers/generated/optuna.samplers.TPESampler.html) y [MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html).

Objetivo A: minimizar costo económico medio de validation, con contrato de servicio ya fijado. Reportar seguridad por separado. No seleccionar A por reward con PF, ni reutilizar test para desempatar.

## G. Propuesta de Optuna para Lagrangian SAC

Implementación a desarrollar y validar. Propuesta mínima: críticos de costo separados, costos almacenados crudos en replay y multiplicadores externos al reward económico, para que cambiar λ no deje penalizaciones obsoletas dentro del buffer.

Para controlar las mismas tasas que se evalúan:
\[
c^V_t=\mathbf{1}\{\exists(i,\phi)\text{ LV}:V_{i\phi,t}\notin[.95,1.05]\},
\quad
c^{PF}_t=\mathbf{1}\{\exists\phi:PF_{\phi,t}<.90\},
\]
\[
J_k=\mathbb{E}\left[\frac1{168}\sum_{t=0}^{167}c^k_t\right]\le d_k,
\quad
\lambda^{k}_{m+1}=\operatorname{clip}
\left(\lambda^{k}_{m}+\eta_\lambda(\widehat J_{k,m}-d_k),0,\lambda_{max}\right).
\]

m es la actualización dual, no cada consulta del solver. J y budgets usan fracción de pasos de un episodio completo, sin descuento. Si el crítico usa retorno descontado, se necesita reconciliar explícitamente su escala/objetivo con esta estimación episódica; no comparar directamente un Q descontado con d=.01.

El objetivo de política combina reward económico/entropía y críticos de costo ponderados por λ. Magnitudes de violación se registran además de los indicadores. Una penalización continua puede facilitar aprendizaje, pero si sustituye al indicador ya no controla exactamente la tasa anterior; debe declararse y justificar su correspondencia.

| Hiperparámetro | Rango propuesto | Condición |
| --- | --- | --- |
| Parámetros SAC compartidos | learning_rate 1e−4–1e−3; arquitectura 2×128/2×256; batch 128/256; gamma .99/.995/.999 | Mismas opciones/budget de A |
| Críticos de costo | Una salida o crítico separado por costo V/PF; arquitectura igual al crítico económico | Estructura fija, verificable |
| Learning rate crítico de costo | Igual al learning_rate del trial | Evitar más dimensiones al inicio |
| eta_lambda | 1e−3 a 1e−1, log | Aplicable SOLO a costos normalizados como J |
| lambda_0 por restricción | 0, .1 o 1 | Categórico |
| lambda_max | 10 | Fijo inicial; evaluar saturación; extensión 10/100 solo si se justifica |
| Actualización dual | Cada 10 episodios, usando promedio de esos episodios | Fija; conservar λ entre resets y reiniciar por trial |
| Suavizado J | Media del bloque de 10 episodios | Fijo; no añadir PID inicialmente |
| Presupuesto d_V, d_PF | Propuesta inicial .01 cada uno | Parámetros científicos a acordar, **no** optimizar con Optuna |
| Coeficiente PF fijo del ambiente | 0 | Evitar doble penalización |
| K_P, K_I, K_D | No aplican al primer diseño dual | No etiquetar como PID |
| Timesteps/trials/validation | Iguales a A | 100800; 20 ampliables a 30; 20 escenarios por evaluación |

λ_max depende de la escala económica elegida: fijar esa escala y registrar saturación antes de interpretar factibilidad. No se garantiza que 10 sea suficiente. Budgets .01 significan tasas promedio, no seguridad dura en todas las horas ni todos los episodios; distinguir eso en el artículo. Targets de servicio/energía terminal deben estar resueltos de manera común antes de añadir estos constraints eléctricos.

Objetivo B: elegir entre configuraciones factibles en validation por menor costo económico; si ninguna satisface budgets, declarar la falta de factibilidad y revisar el diseño sin consultar test. No ajustar budgets, umbrales eléctricos ni λ_pf fijo para favorecer resultados.

## H. Estimación de costo computacional

Solo se midió simulación; no se midió forward/backward SAC, GPU, escritura de replay, callbacks ni costo de críticos adicionales. Por tanto:
\[
T_{total}\approx N_{train}\,t_{env}
 +N_{updates}\,t_{update}
 +N_{eval}\,t_{env}
 +T_{inferencia/log/IO},
\quad t_{env}\approx.017\text{–}.020\;s.
\]

| Presupuesto propuesto | Pasos de ambiente | Tiempo de simulación estimado |
| --- | ---: | ---: |
| Episodio | 168 | 2.9–3.4 s |
| Un trial, entrenamiento | 100800 | 28.6–33.6 min |
| Validation por trial: 6×20 episodios | 20160 | 5.7–6.7 min |
| Un trial completo sin aprendizaje neuronal | 120960 | 34.3–40.3 min |
| 20 trials de un algoritmo | 2419200 | 11.4–13.4 h |
| 30 trials de un algoritmo | 3628800 | 17.1–20.2 h |
| A+B, 20 trials cada uno | 4838400 | 22.8–26.9 h |
| A+B, 30 trials cada uno | 7257600 | 34.3–40.3 h |
| Confirmation, una seed de 504000 pasos | 504000 | 2.38–2.80 h |
| Confirmation, 5 seeds, un algoritmo | 2520000 | 11.9–14.0 h |
| Confirmation, ambos algoritmos, 5 seeds | 5040000 | 23.8–28.0 h |
| Test: 200 escenarios, una seed/política | 33600 | 9.5–11.2 min |
| Test: dos SAC ×5 seeds ×200 escenarios | 336000 | 95.2–112 min |

No incluye validation adicional de confirmation, regla determinista, reentrenamientos de finalistas ni posibles episodios nominales emparejados extra. Pruning podría reducir tiempo; actualizaciones neuronales lo aumentarán. Estas cifras no son promesas de wall time total ni de convergencia. El siguiente benchmark de entrenamiento deberá ser corto, autorizado después de resolver bloqueos, para estimar t_update y uso CPU/GPU antes de reservar días de cómputo.

## I. Checklist para convertir el ambiente en un benchmark publicable

- [x] Verificar observación/acción, fases, reward y signos efectivos.
- [x] Verificar reset/step, fin de episodio y reproducibilidad explícita en un stack.
- [x] Identificar protecciones locales y costos ya presentes.
- [x] Ejecutar pruebas breves y medir el cuello computacional.
- [ ] Resolver servicio V2G y energía terminal.
- [ ] Corregir/justificar neutros, calendario y parámetros eléctricos.
- [ ] Publicar límites de seguridad, costos y budgets inequívocos.
- [ ] Separar reward económico de costos eléctricos en ambos SAC.
- [ ] Reconstruir metadatos de fechas/edificios y trazabilidad solar.
- [ ] Congelar split por semanas base y manifiestos de escenarios.
- [ ] Documentar forecast sintético y evitar acceso privilegiado a info.
- [ ] Garantizar semillas de campaña, eventos, acciones y aprendizaje.
- [ ] Fijar dominio eléctrico y tratamiento de no convergencia/residual.
- [ ] Añadir métricas de fase, servicio, intervención y agregación episódica.
- [ ] Crear paquete, requirements/lock, README y ejemplo mínimo.
- [ ] Declarar licencias/atribuciones de código, datos y parámetros.
- [ ] Ejecutar CI en entorno limpio y revisar suite completa.
- [ ] Implementar regla de servicio, SAC y SAC dual con pruebas adecuadas.
- [ ] Separar tuning, confirmation y evaluación final ciega.
- [ ] Publicar todas las seeds, curvas, configuraciones y limitaciones; ninguna selección por test.

Huellas SHA-256 de pools comprobadas (copias CODIGO=DATA_DIR):
- office: `528bf364aef0719d57ac4acad2c8ad520e04a73affc86746486c339c53b80fb5`
- warehouse1: `54afad00eef6f528183e22bb8a354dc89bcc2501c45f72a0d20a650208381dcc`
- warehouse2: `159ebd1d2725e378de53cbe1a14d54c5b75e91d45d2ab227e5a375916bb9ea44`
- solar: `161c01621b0de8413e1e0a8c2636c757a785457750da0f9fb60761d4fef198df`

Huellas de referencia de las fuentes principales:
- Ambiente_industrial.py: `3f242b08ae7820f822a18678204a5e13a08c83a322e9eb0db6bb75a1f8517f22`
- opendss_easy.py: `8682749f6aa1276f50c3c77fde0c58ae505eb19950afad9f3a969f7bd0eea32a`

## J. Archivos exactos que habría que modificar en una siguiente fase

Solo listado de cambios sugeridos; **ninguno realizado**.

| Archivo existente o nuevo propuesto | Cambio sugerido |
| --- | --- |
| [Ambiente_industrial.py](<env/Ambiente_industrial.py>) | Contrato de escenario/split, seeds, servicio y terminal; costos eléctricos separados; métricas; configuración serializable; comentarios |
| [opendss_easy.py](<env/opendss_easy.py>) | Conexiones de neutro elegidas explícitamente, C/ampacidad documentadas, corrientes ABC, aislamiento de contexto, dominio/residual; optimización solo tras equivalencia |
| [oficinas.py](<env/oficinas.py>) | IDs/fechas/splits permitidos, política de escalado/forecast, parámetros efectivos y APIs legacy |
| [PV_plant.py](<env/PV_plant.py>) | Selección por manifiesto, trazabilidad solar, supuesto de forecast y parámetros públicos coherentes |
| [price_sampler.py](<env/price_sampler.py>) | Procedencia tarifaria, contrato económico y generación por escenario reproducible |
| [v2g_bus_5.py](<env/v2g_bus_5.py>) | Exponer evento/energía de salida y llegada; opciones de Q y escalado si se generalizan |
| [battery.py](<env/battery.py>) | Solo validación de parámetros o soporte contable terminal si el contrato lo requiere; preservar dinámica y signos |
| [vehiculos_Zoe.py](<env/vehiculos_Zoe.py>) | Calendario coherente con escenarios fechados y telemetría de servicio |
| [extreme_events.py](<env/extreme_events.py>) | Solo ajustes necesarios de esquema/manifiesto de eventos; corregir el uso de RNG principalmente en E |
| `extracciondatos_edificio_oficinas.ipynb` (historical reference; not distributed) | Exportar IDs/fechas y documentar descarte/calidad; conservar original |
| `exploracion warehouses.ipynb` (historical reference; not distributed) | Alinear semanas ISO y exportar metadatos de filas retenidas |
| `prueba camion bus5.ipynb` (historical reference; not distributed) | Actualizar import/API; revisar los otros notebooks de ejemplo |
| [test_environment_step.py](<tests/test_environment_step.py>) | Servicio/terminal, costos, fallos y métricas del nuevo contrato |
| [test_environment_reset_2c.py](<tests/test_environment_reset_2c.py>) | Split, selección explícita, constructor seed y secuencias auto-reset |
| `test_extreme_event_random_integration.py` (historical reference; not distributed) | Reproducción de campañas con reset(None) |
| [test_transformer_production.py](<tests/test_transformer_production.py>) | Verificar NodeOrder/NodeRef, no solo valores Rneut |
| [test_reward.py](<tests/test_reward.py>) | Separación de economía y constraints sin doble penalización |
| [test_electrical_instrumentation.py](<tests/test_electrical_instrumentation.py>) | Corrientes por fase, residuales y umbrales acordados |
| `transformer_episode_diagnostics.py` (historical reference; not distributed) | Migrar evaluación operacional a step; mantener campañas históricas identificadas |
| `pyproject.toml` (historical reference; not distributed) — nuevo | Paquete y dependencias/versiones |
| `requirements-lock.txt` (historical reference; not distributed) — nuevo | Stack reproducible por plataforma |
| `README.md` (historical reference; not distributed) — nuevo | Instalación, uso, hipótesis, datos y reproducción |
| `LICENSE` (historical reference; not distributed) — nuevo | Licencia elegida para autoría del ambiente |
| `DATA_PROVENANCE.md` (historical reference; not distributed) — nuevo | Origen de pools, fechas, licencias, hashes y parámetros eléctricos |
| `prepare_scenarios.py` (historical reference; not distributed) — nuevo | Generación trazable, alineación y manifiestos inmutables de splits |
| [scenario_splits.json](<data/scenario_splits.json>) — nuevo | IDs de semanas y escenarios reservados |
| `baselines.py` (historical reference; not distributed) — nuevo | Regla determinista con servicio común |
| [train_sac.py](<scripts/train_sac.py>) — nuevo | Entrenamiento SAC, seeds, callbacks y checkpoints |
| `lagrangian_sac.py` (historical reference; not distributed) — nuevo | Replay de costos, críticos y actualización dual |
| `tune_sac.py` (historical reference; not distributed) — nuevo | Estudios A/B, presupuestos y selección solo con validation |
| `evaluate_agents.py` (historical reference; not distributed) — nuevo | Métricas emparejadas, seeds, fallos, agregación y test ciego |
