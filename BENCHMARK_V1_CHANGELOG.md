> Distribution note (2026-09-09): historical technical document, with personal paths sanitized. The original audit is subordinate to BENCHMARK_V1_CHANGELOG.md. References to excluded artifacts are historical. See the snapshot addendum in the changelog for current packaging differences.

# BENCHMARK_V1_CHANGELOG

Fecha: 2026-09-08. Estado: implementación estabilizada y validada; falta fijar explícitamente el coeficiente científico `c_unserved` antes de iniciar un episodio de entrenamiento.

Preparación realizada a partir de `AUDITORIA_AMBIENTE_TRIFASICO.md`, que se conserva sin cambios como evidencia histórica. Se sigue el alcance actual, incluidas sus decisiones sobre PF, BESS, neutro y forecasts. No se entrenó SAC, no se implementó Lagrangian SAC y no se ejecutaron Optuna, conformal prediction ni proyección QP. No se rediseñó ni optimizó el flujo eléctrico.

## 1. Archivos modificados y creados

| Archivo | Cambio |
| --- | --- |
| [Ambiente_industrial.py](<env/Ambiente_industrial.py>) | Split y escenario, c_unserved explícito, costo de salida, métricas y documentación de neutro. |
| [scenario_catalog.py](<env/scenario_catalog.py>) | Nuevo: identidad, aislamiento y verificación de hashes antes de cargar el split. |
| [prepare_benchmark_v1_data.py](<scripts/prepare_benchmark_v1_data.py>) | Nuevo: reconstrucción desde timestamps y generación determinista de derivados. |
| [benchmark_metrics.py](<env/benchmark_metrics.py>) | Nuevo: eventos de salida y agregación descriptiva. |
| [scenario_splits.json](<data/scenario_splits.json>) | Nuevo: fuentes, filtros, descartes, fechas, hashes y composiciones exactas. |
| [test_benchmark_v1.py](<tests/test_benchmark_v1.py>) | Nuevo: 18 pruebas de aceptación. |
| [run_benchmark_v1_checks.py](<scripts/run_benchmark_v1_checks.py>) | Nuevo: ejecutor nominal que excluye las pruebas de eventos aplazadas. |
| `benchmark_v1_checks.json` (historical reference; not distributed); `benchmark_v1_checks.log` (historical reference; not distributed) | Resultados individuales de 140 pruebas. |
| `benchmark_v1_integrity.json` (historical reference; not distributed) | Evidencias de integridad, reconstrucción idéntica y comparación AST. |

Pruebas existentes adaptadas: declaran `TEST_C_UNSERVED=1.0` como fixture arbitraria, sin convertirlo en valor científico. La comparación de telemetría excluye únicamente `episode_runtime`; el test del transformador comprueba su NodeOrder y el número de nodos. Los tests de eventos reciben la adaptación de interfaz, pero no se ejecutan en esta fase.

- [test_electrical_instrumentation.py](<tests/test_electrical_instrumentation.py>)
- [test_electrical_snapshot_2e.py](<tests/test_electrical_snapshot_2e.py>)
- [test_environment_reset_2c.py](<tests/test_environment_reset_2c.py>)
- [test_environment_step.py](<tests/test_environment_step.py>)
- [test_episode_render.py](<tests/test_episode_render.py>)
- [test_exogenous_realization_layers.py](<tests/test_exogenous_realization_layers.py>)
- `test_extreme_event_load_integration.py` (historical reference; not distributed)
- `test_extreme_event_pv_integration.py` (historical reference; not distributed)
- `test_extreme_event_random_integration.py` (historical reference; not distributed)
- [test_load_telemetry.py](<tests/test_load_telemetry.py>)
- [test_opendss_easy_1c_b.py](<tests/test_opendss_easy_1c_b.py>)
- [test_price_sampler.py](<tests/test_price_sampler.py>)
- [test_pv_action_decoder_2d.py](<tests/test_pv_action_decoder_2d.py>)
- [test_pv_spatial_variation_2d1.py](<tests/test_pv_spatial_variation_2d1.py>)
- [test_reward.py](<tests/test_reward.py>)
- [test_temporal_contract_2b.py](<tests/test_temporal_contract_2b.py>)
- [test_transformer_production.py](<tests/test_transformer_production.py>)
- [test_vminpu_constant_power.py](<tests/test_vminpu_constant_power.py>)

Se conservan byte a byte los ocho módulos físicos/exógenos listados en la sección 5. No se crea automáticamente un commit ni un tag.

## 2. Datasets derivados creados

Cada NPZ contiene `weeks`, una matriz float64 de n×168, y `base_ids`, un vector de texto sin objetos pickle. Los valores horarios originales se conservan: el escalado y la generación estocástica existentes se aplican después en los samplers. Ningún original se sobrescribe.

| Split | Fuente | Semanas | Archivo |
| --- | --- | --- | --- |
| train | office | 69 | [office_weeks_pool.npz](<data/train/office_weeks_pool.npz>) |
| train | wh1 | 55 | [wh1_weeks_pool.npz](<data/train/wh1_weeks_pool.npz>) |
| train | wh2 | 85 | [wh2_weeks_pool.npz](<data/train/wh2_weeks_pool.npz>) |
| train | solar | 68 | [solar_weeks_pool.npz](<data/train/solar_weeks_pool.npz>) |
| optuna_validation | office | 6 | [office_weeks_pool.npz](<data/optuna_validation/office_weeks_pool.npz>) |
| optuna_validation | wh1 | 6 | [wh1_weeks_pool.npz](<data/optuna_validation/wh1_weeks_pool.npz>) |
| optuna_validation | wh2 | 6 | [wh2_weeks_pool.npz](<data/optuna_validation/wh2_weeks_pool.npz>) |
| optuna_validation | solar | 6 | [solar_weeks_pool.npz](<data/optuna_validation/solar_weeks_pool.npz>) |
| final_test | office | 6 | `office_weeks_pool.npz` (historical reference; not distributed) |
| final_test | wh1 | 6 | `wh1_weeks_pool.npz` (historical reference; not distributed) |
| final_test | wh2 | 6 | `wh2_weeks_pool.npz` (historical reference; not distributed) |
| final_test | solar | 6 | `solar_weeks_pool.npz` (historical reference; not distributed) |

## 3. Procedimiento de realineación temporal

Fuente fechada: `electricity_cleaned.csv` (historical reference; not distributed). Columnas originales: `Panther_office_Karla`, `Hog_warehouse_Louise`, `Rat_warehouse_Shari` y `timestamp`. Cobertura: 2016-01-01 00:00 a 2017-12-31 23:00, 17.544 horas. Los timestamps son naive; no se les asigna una zona horaria inventada.

El constructor ordena timestamps, rechaza duplicados y agrupa por períodos `W-SUN` que comienzan el lunes. Solo admite exactamente 168 timestamps distintos e idénticos a la secuencia horaria lunes 00:00 → domingo 23:00, con valores finitos y no negativos. No interpola, imputa ni corta bloques consecutivos desde el viernes.

Para oficina se conserva la cobertura del extractor histórico desde 2016-05-23; no se añaden retrospectivamente semanas que estaban fuera del pool auditado. Se eliminan sus tres semanas no finitas. Para bodegas se conservan los filtros originales, aplicados tras realinear: máximo 15 % de ceros y suma semanal ≥20 en las unidades originales de la serie horaria.

| Fuente | Grupos/perfiles examinados | Válidos | Descartados |
| --- | --- | --- | --- |
| office | 105 | 81 | 24 |
| wh1 | 105 | 67 | 38 |
| wh2 | 105 | 97 | 8 |
| solar | 80 | 80 | 0 |

Motivos de descarte. Un mismo grupo puede tener más de un motivo; estos conteos no se suman como semanas únicas:

| Fuente | Motivo | Grupos |
| --- | --- | --- |
| office | not_168_unique_consecutive_hours_monday_to_sunday | 1 |
| office | outside_legacy_office_coverage_start_2016_05_23 | 21 |
| office | nonfinite_values | 3 |
| wh1 | not_168_unique_consecutive_hours_monday_to_sunday | 1 |
| wh1 | nonfinite_values | 37 |
| wh2 | not_168_unique_consecutive_hours_monday_to_sunday | 1 |
| wh2 | nonfinite_values | 7 |

En estos datos no se descartan semanas adicionales exclusivamente por exceso de ceros o suma insuficiente. El manifiesto registra fechas y todos los motivos de cada grupo descartado. `aligned_pool_index` identifica la posición de una carga en su pool válido realineado; en solar `original_index` es la fila del NPZ original. `week_indices` de reset son índices locales del derivado seleccionado; para trazabilidad se usan `base_week_ids`.

Los 80 perfiles solares se mantienen como perfiles genéricos de 168 horas, sin inicio/fin reales, con índices originales y hashes. Se emplean como coordenadas semanales del episodio, sin afirmar que correspondan a las fechas de las cargas. El calendario de vehículos sigue lunes→domingo.

## 4. Composición exacta de los tres splits

La separación precede a ruido, PF, reparto ABC, precios, SOC y forecasts. Cambiar seed no mueve un perfil a otro split. El catálogo comprueba partición exhaustiva y disjunta por ID, hash y fecha; una fecha tampoco puede cruzar splits entre las tres fuentes fechadas.

Hay 54 lunes completos válidos comunes a las tres cargas. Se reservan los últimos 12 del conjunto común: seis anteriores para optuna_validation y seis posteriores para final_test. En solar se reservan las últimas 12 filas válidas originales, con la misma división 6+6, sin atribuirles cronología. La selección no utiliza recompensas, costos ni evaluaciones de políticas.

Todas las demás semanas válidas quedan en train: 69 de oficina, 55 de bodega 1, 85 de bodega 2 y 68 solares. Son cantidades por fuente, no 277 semanas industriales sincronizadas. Quedan 42 lunes comunes en train, además de semanas válidas disponibles solo en algunas fuentes. Train combina sus fuentes independientemente. Cada escenario de referencia de evaluación empareja el mismo lunes en las tres cargas con un perfil solar genérico fijo.

El split NO es estrictamente cronológico: algunas semanas no comunes asignadas a train son posteriores a fechas reservadas. Esta separación de bases conserva todas las semanas restantes, como se solicitó; no constituye un experimento de generalización temporal prospectiva.

| Etiqueta | Lunes cargas | Domingo cargas | Perfil solar |
| --- | --- | --- | --- |
| optuna_validation_01 | 2017-08-14 | 2017-08-20 | solar-generic-0068 |
| optuna_validation_02 | 2017-08-21 | 2017-08-27 | solar-generic-0069 |
| optuna_validation_03 | 2017-08-28 | 2017-09-03 | solar-generic-0070 |
| optuna_validation_04 | 2017-09-04 | 2017-09-10 | solar-generic-0071 |
| optuna_validation_05 | 2017-09-11 | 2017-09-17 | solar-generic-0072 |
| optuna_validation_06 | 2017-09-18 | 2017-09-24 | solar-generic-0073 |
| final_test_01 | 2017-09-25 | 2017-10-01 | solar-generic-0074 |
| final_test_02 | 2017-10-02 | 2017-10-08 | solar-generic-0075 |
| final_test_03 | 2017-10-09 | 2017-10-15 | solar-generic-0076 |
| final_test_04 | 2017-10-16 | 2017-10-22 | solar-generic-0077 |
| final_test_05 | 2017-10-23 | 2017-10-29 | solar-generic-0078 |
| final_test_06 | 2017-11-20 | 2017-11-26 | solar-generic-0079 |

Inventario exhaustivo. Cada ID identifica exactamente una fila base del manifiesto:

**train**

- **office (69)**: `office-2016-W21`, `office-2016-W22`, `office-2016-W23`, `office-2016-W24`, `office-2016-W25`, `office-2016-W26`, `office-2016-W27`, `office-2016-W28`, `office-2016-W29`, `office-2016-W30`, `office-2016-W31`, `office-2016-W32`, `office-2016-W33`, `office-2016-W34`, `office-2016-W35`, `office-2016-W36`, `office-2016-W37`, `office-2016-W38`, `office-2016-W39`, `office-2016-W40`, `office-2016-W41`, `office-2016-W42`, `office-2016-W43`, `office-2016-W44`, `office-2016-W45`, `office-2016-W46`, `office-2016-W47`, `office-2016-W48`, `office-2016-W49`, `office-2016-W50`, `office-2016-W51`, `office-2016-W52`, `office-2017-W01`, `office-2017-W02`, `office-2017-W03`, `office-2017-W04`, `office-2017-W05`, `office-2017-W06`, `office-2017-W08`, `office-2017-W09`, `office-2017-W11`, `office-2017-W12`, `office-2017-W13`, `office-2017-W14`, `office-2017-W15`, `office-2017-W16`, `office-2017-W17`, `office-2017-W18`, `office-2017-W19`, `office-2017-W20`, `office-2017-W21`, `office-2017-W22`, `office-2017-W23`, `office-2017-W24`, `office-2017-W26`, `office-2017-W27`, `office-2017-W28`, `office-2017-W29`, `office-2017-W30`, `office-2017-W31`, `office-2017-W32`, `office-2017-W44`, `office-2017-W45`, `office-2017-W46`, `office-2017-W48`, `office-2017-W49`, `office-2017-W50`, `office-2017-W51`, `office-2017-W52`.
- **wh1 (55)**: `wh1-2016-W01`, `wh1-2016-W14`, `wh1-2016-W15`, `wh1-2016-W16`, `wh1-2016-W17`, `wh1-2016-W18`, `wh1-2016-W19`, `wh1-2016-W20`, `wh1-2016-W21`, `wh1-2016-W22`, `wh1-2016-W23`, `wh1-2016-W24`, `wh1-2016-W25`, `wh1-2016-W26`, `wh1-2016-W27`, `wh1-2016-W28`, `wh1-2016-W29`, `wh1-2016-W30`, `wh1-2016-W31`, `wh1-2016-W32`, `wh1-2016-W33`, `wh1-2016-W34`, `wh1-2016-W35`, `wh1-2016-W36`, `wh1-2016-W37`, `wh1-2016-W38`, `wh1-2016-W39`, `wh1-2016-W40`, `wh1-2016-W41`, `wh1-2016-W42`, `wh1-2016-W43`, `wh1-2016-W44`, `wh1-2016-W45`, `wh1-2016-W46`, `wh1-2016-W47`, `wh1-2016-W49`, `wh1-2016-W50`, `wh1-2016-W51`, `wh1-2017-W12`, `wh1-2017-W13`, `wh1-2017-W18`, `wh1-2017-W19`, `wh1-2017-W20`, `wh1-2017-W21`, `wh1-2017-W22`, `wh1-2017-W23`, `wh1-2017-W24`, `wh1-2017-W25`, `wh1-2017-W26`, `wh1-2017-W27`, `wh1-2017-W28`, `wh1-2017-W30`, `wh1-2017-W31`, `wh1-2017-W32`, `wh1-2017-W44`.
- **wh2 (85)**: `wh2-2016-W01`, `wh2-2016-W02`, `wh2-2016-W03`, `wh2-2016-W04`, `wh2-2016-W05`, `wh2-2016-W06`, `wh2-2016-W07`, `wh2-2016-W08`, `wh2-2016-W09`, `wh2-2016-W11`, `wh2-2016-W12`, `wh2-2016-W13`, `wh2-2016-W14`, `wh2-2016-W15`, `wh2-2016-W16`, `wh2-2016-W17`, `wh2-2016-W18`, `wh2-2016-W19`, `wh2-2016-W20`, `wh2-2016-W22`, `wh2-2016-W23`, `wh2-2016-W24`, `wh2-2016-W25`, `wh2-2016-W26`, `wh2-2016-W27`, `wh2-2016-W28`, `wh2-2016-W29`, `wh2-2016-W30`, `wh2-2016-W31`, `wh2-2016-W32`, `wh2-2016-W33`, `wh2-2016-W34`, `wh2-2016-W35`, `wh2-2016-W36`, `wh2-2016-W37`, `wh2-2016-W38`, `wh2-2016-W39`, `wh2-2016-W40`, `wh2-2016-W41`, `wh2-2016-W42`, `wh2-2016-W43`, `wh2-2016-W45`, `wh2-2016-W46`, `wh2-2016-W47`, `wh2-2016-W48`, `wh2-2016-W49`, `wh2-2016-W50`, `wh2-2016-W51`, `wh2-2016-W52`, `wh2-2017-W01`, `wh2-2017-W02`, `wh2-2017-W03`, `wh2-2017-W04`, `wh2-2017-W05`, `wh2-2017-W06`, `wh2-2017-W07`, `wh2-2017-W08`, `wh2-2017-W09`, `wh2-2017-W11`, `wh2-2017-W12`, `wh2-2017-W13`, `wh2-2017-W14`, `wh2-2017-W15`, `wh2-2017-W16`, `wh2-2017-W17`, `wh2-2017-W18`, `wh2-2017-W19`, `wh2-2017-W20`, `wh2-2017-W21`, `wh2-2017-W22`, `wh2-2017-W23`, `wh2-2017-W24`, `wh2-2017-W27`, `wh2-2017-W28`, `wh2-2017-W29`, `wh2-2017-W30`, `wh2-2017-W31`, `wh2-2017-W32`, `wh2-2017-W45`, `wh2-2017-W46`, `wh2-2017-W48`, `wh2-2017-W49`, `wh2-2017-W50`, `wh2-2017-W51`, `wh2-2017-W52`.
- **solar (68)**: `solar-generic-0000`, `solar-generic-0001`, `solar-generic-0002`, `solar-generic-0003`, `solar-generic-0004`, `solar-generic-0005`, `solar-generic-0006`, `solar-generic-0007`, `solar-generic-0008`, `solar-generic-0009`, `solar-generic-0010`, `solar-generic-0011`, `solar-generic-0012`, `solar-generic-0013`, `solar-generic-0014`, `solar-generic-0015`, `solar-generic-0016`, `solar-generic-0017`, `solar-generic-0018`, `solar-generic-0019`, `solar-generic-0020`, `solar-generic-0021`, `solar-generic-0022`, `solar-generic-0023`, `solar-generic-0024`, `solar-generic-0025`, `solar-generic-0026`, `solar-generic-0027`, `solar-generic-0028`, `solar-generic-0029`, `solar-generic-0030`, `solar-generic-0031`, `solar-generic-0032`, `solar-generic-0033`, `solar-generic-0034`, `solar-generic-0035`, `solar-generic-0036`, `solar-generic-0037`, `solar-generic-0038`, `solar-generic-0039`, `solar-generic-0040`, `solar-generic-0041`, `solar-generic-0042`, `solar-generic-0043`, `solar-generic-0044`, `solar-generic-0045`, `solar-generic-0046`, `solar-generic-0047`, `solar-generic-0048`, `solar-generic-0049`, `solar-generic-0050`, `solar-generic-0051`, `solar-generic-0052`, `solar-generic-0053`, `solar-generic-0054`, `solar-generic-0055`, `solar-generic-0056`, `solar-generic-0057`, `solar-generic-0058`, `solar-generic-0059`, `solar-generic-0060`, `solar-generic-0061`, `solar-generic-0062`, `solar-generic-0063`, `solar-generic-0064`, `solar-generic-0065`, `solar-generic-0066`, `solar-generic-0067`.

**optuna_validation**

- **office (6)**: `office-2017-W33`, `office-2017-W34`, `office-2017-W35`, `office-2017-W36`, `office-2017-W37`, `office-2017-W38`.
- **wh1 (6)**: `wh1-2017-W33`, `wh1-2017-W34`, `wh1-2017-W35`, `wh1-2017-W36`, `wh1-2017-W37`, `wh1-2017-W38`.
- **wh2 (6)**: `wh2-2017-W33`, `wh2-2017-W34`, `wh2-2017-W35`, `wh2-2017-W36`, `wh2-2017-W37`, `wh2-2017-W38`.
- **solar (6)**: `solar-generic-0068`, `solar-generic-0069`, `solar-generic-0070`, `solar-generic-0071`, `solar-generic-0072`, `solar-generic-0073`.

**final_test**

- **office (6)**: `office-2017-W39`, `office-2017-W40`, `office-2017-W41`, `office-2017-W42`, `office-2017-W43`, `office-2017-W47`.
- **wh1 (6)**: `wh1-2017-W39`, `wh1-2017-W40`, `wh1-2017-W41`, `wh1-2017-W42`, `wh1-2017-W43`, `wh1-2017-W47`.
- **wh2 (6)**: `wh2-2017-W39`, `wh2-2017-W40`, `wh2-2017-W41`, `wh2-2017-W42`, `wh2-2017-W43`, `wh2-2017-W47`.
- **solar (6)**: `solar-generic-0074`, `solar-generic-0075`, `solar-generic-0076`, `solar-generic-0077`, `solar-generic-0078`, `solar-generic-0079`.

El ambiente abre únicamente los cuatro archivos de su split. `split` es de solo lectura. Un ID de otro split o manifiesto se rechaza antes de modificar el episodio. `ScenarioCatalog.reference_scenario_ids(split)` devuelve las seis composiciones de evaluación; para evaluar exhaustivamente se recorren todas, sin confiar en seis sorteos aleatorios.

El catálogo acepta otras combinaciones explícitas válidas del mismo split, pero el protocolo de evaluación de seis escenarios debe utilizar la lista de referencia fija. La selección aleatoria de un ambiente de evaluación también se limita a esa lista.

El futuro controlador de Optuna debe construir únicamente train y optuna_validation. final_test no participará en selección de coeficientes, hiperparámetros, poda, checkpoints ni estadísticas de normalización. En esta fase no se ejecutaron episodios de final_test ni evaluaciones de políticas en holdouts: solo se inspeccionó la integridad de sus datos y manifiesto.

## 5. Hashes y estado verificable

Commit de referencia del código auditado: `863e699941517ea632cc93a22b24e9802998a39b`. SHA-256 del manifiesto: `0db4a1ccbceec4833d6c94a131efe9e5060b6e09b43198522563269ff2c0f542`. Sus primeros 12 caracteres forman parte del scenario_id; el hash completo se expone en `reset.info["scenario_manifest_sha256"]`.

| Fuente original | SHA-256 |
| --- | --- |
| DATOS NATURE\buds-lab\building-data-genome-project-2-v1.0\buds-lab-building-data-genome-project-2-3d0cbaf\data\meters\cleaned\electricity_cleaned.csv | b01056f4ceec1992ce019297e674d6d6ecd0240e50a2aaeb6f958c65efdc282d |
| office_weeks_pool.npz | 528bf364aef0719d57ac4acad2c8ad520e04a73affc86746486c339c53b80fb5 |
| warehouse1_weeks_pool.npz | 54afad00eef6f528183e22bb8a354dc89bcc2501c45f72a0d20a650208381dcc |
| warehouse2_weeks_pool.npz | 159ebd1d2725e378de53cbe1a14d54c5b75e91d45d2ab227e5a375916bb9ea44 |
| solar_weeks_pool.npz | 161c01621b0de8413e1e0a8c2636c757a785457750da0f9fb60761d4fef198df |

Los cuatro NPZ originales conservan los hashes auditados tanto en CODIGO como en el directorio canónico. El constructor únicamente lee el CSV original y el NPZ solar canónico.

| Derivado | SHA-256 |
| --- | --- |
| benchmark_v1_data/train/office_weeks_pool.npz | d1e9a8938b24ca06e102eb5a2b1102d95909214ffdec51f7c86a463d8df2077c |
| benchmark_v1_data/train/wh1_weeks_pool.npz | 052b15879177b3241ca1af23f86fa5d50bd962bac12af727458ee8292f11b6b7 |
| benchmark_v1_data/train/wh2_weeks_pool.npz | 2af32b0cdfe818f55086a57fa56cd7ba559c2b8beace8147e922cfa57d698ff3 |
| benchmark_v1_data/train/solar_weeks_pool.npz | 07cde2e9d29184583231c8c02ec262d0cf331418c64e3e0ad805c4410cbf27c8 |
| benchmark_v1_data/optuna_validation/office_weeks_pool.npz | a4c943b93d194978d8e75badf150f616c5dc67d74706913e3159f40f3103db62 |
| benchmark_v1_data/optuna_validation/wh1_weeks_pool.npz | a8544196006546855a0d05aeb31e14e7e289ecb26b69c2684cb1e903fcca2963 |
| benchmark_v1_data/optuna_validation/wh2_weeks_pool.npz | 1cc3c7c56d54ff94352b7b4e82b06564371b88cf8bcaab978a273d60787fd84b |
| benchmark_v1_data/optuna_validation/solar_weeks_pool.npz | c7579f37e4715156400704760eeb4e53f1ed2f92a5b4594fbb08f13aaf9b17e6 |
| benchmark_v1_data/final_test/office_weeks_pool.npz | 5d970ad721a0012de992422de4ae116b9091e3621138588e176579f38d8d7c65 |
| benchmark_v1_data/final_test/wh1_weeks_pool.npz | afb482e9a4780752e4b1bde4c3700092fe0b70232e15ba3b73737b76f98ca741 |
| benchmark_v1_data/final_test/wh2_weeks_pool.npz | 13299c459149d31bc35248f22c4798d794c51e934cb2fde798efa5d3f53165ea |
| benchmark_v1_data/final_test/solar_weeks_pool.npz | 312609216a4c917a5a4223eacf861b3e03160edc29956a48fe560c2619fcf7e1 |

Cada fila también tiene hash SHA-256 de los valores `<f8`. Una reconstrucción completa produjo exactamente los mismos bytes en manifiesto y doce NPZ, gracias al orden y timestamps ZIP fijos. El catálogo verifica hash de archivo, forma, IDs y hash individual al cargar el pool; si falla, no vuelve a los datos originales.

| Código del benchmark | SHA-256 |
| --- | --- |
| Ambiente_industrial.py | 90a265970a8ab97e463f84ff27a81828af766df3333796cc8cb3d37e41ce43a1 |
| scenario_catalog.py | 28b4bbc8aee3337a0160db002977c8304d73ae44856c402e46f468ec07c2ebbf |
| prepare_benchmark_v1_data.py | e66da66ba6d582cf1bb42176e1c12005513d75d930f53dfb1b41928ec4ee48d0 |
| benchmark_metrics.py | b1f641d42b53128e5e04dd6dea923e191a511f0d63f3e77ed0108f15a46f73b6 |
| run_benchmark_v1_checks.py | 98f56cf956c6d9e774672807d8be4cb7853275d336e3fa4ff491b7c838e16fd7 |

| Módulo intacto | SHA-256 |
| --- | --- |
| battery.py | f5a5f4a1be43574e02721f1eab84fb2987584efdff2c85c8b85c322c30c542a7 |
| extreme_events.py | 84b19e27a39b4ae70312456d2ca1337346ae4dc95182571d6bf8090d5dbe3e79 |
| oficinas.py | ae77b9619a39643e04736e6b9993d010996119df3956bd9b5598fe5211d8b65f |
| opendss_easy.py | 8682749f6aa1276f50c3c77fde0c58ae505eb19950afad9f3a969f7bd0eea32a |
| price_sampler.py | d9fc6284d3b0cfe3824d535a9b98276f55d8d4d3525ed8218b7567176f08e590 |
| PV_plant.py | 3d3ae190c1eb9cd38fd519516a262263378ba23398a0e328850a1335ee957277 |
| v2g_bus_5.py | 818f0e078124fbb5bedee03da72b3f1cfe292e1a94cf7db4e169fd4c4e1b89a5 |
| vehiculos_Zoe.py | d90c264857e98fa0d2c4cf7a6f16cb116939ac0e0f7aea5fb497a5a6fe5ddd92 |

La comparación AST, excluyendo docstrings, confirmó 34 métodos preexistentes idénticos, incluidos `_solve_electrical_snapshot`, `_decode_action`, `_get_raw_obs`, `_get_obs`, `_normalize_obs`, construcción de red y preparación horaria. Solo cambian los métodos preexistentes `__init__`, `reset`, `step` y `_compute_reward`; se añaden accesores de split y métricas. La descripción textual del neutro es metadata: no modifica comandos eléctricos.

## 6. Fórmula completa del reward

Para Δt=1 hora, importación positiva y exportación negativa:

```text
C_energy,t = P_grid,t × (price_raw,t / 100) × Δt
C_PF,t = lambda_pf × Δt × Σ_phi max(0, 0.90 − PF_phi,t) × |Q_PCC,phi,t|
E_deficit,j,t = max(0, (0.80 − SOC_departure,j,t) × E_capacity,j)
C_truck,t = Σ_j en salidas(t+1) c_unserved × E_deficit,j,t
reward_t = −C_energy,t − C_PF,t − C_truck,t
```

El precio crudo continúa en centavos de USD/kWh. Exportar genera ingreso con la tarifa existente, igual para compra/venta. El término PF permanece intacto, incluido lambda_pf predeterminado 1.0. PF y Q proceden del snapshot del PCC en orden ABC. No se añade costo de voltaje, VUF, loading ni SOC terminal.

`info` separa `energy_cost_usd`, `pf_penalty_usd`, `truck_departure_penalty_usd` y `reward`. `operating_cost_usd` conserva su papel de alias del costo de energía. `RewardBreakdown.energy_cost_usd` es una propiedad compatible con el campo histórico. La función pura `compute_reward` admite costo de salida cero por defecto para intervalos sin salida; el ambiente requiere c_unserved explícito antes de reset.

`c_unserved` es USD/kWh no servido, finito y positivo, sin valor científico predeterminado: `None`. No es hiperparámetro de Optuna. Los valores 1 y 2 de las pruebas son fixtures para comprobar identidades, no una calibración ni una recomendación científica. El futuro SAC de referencia se denomina **baseline SAC** o **vanilla SAC**, conservando la penalización económica de PF.

## 7. Tratamiento de salidas de camiones

Una salida ocurre si el camión está conectado en t y estará desconectado en t+1 dentro del horizonte. Su SOC de salida es `resolution.soc_next`, tras la acción resuelta para [t,t+1). Primero se obtiene un snapshot convergente, después se calculan eventos y reward sin mutar energía, y luego se confirma la transición: **solve → reward → commit**.

Se conserva byte a byte la dinámica del truck y su clipping existente. No se fuerza llegar a 0.80. Cada camión tiene actualmente 315 kWh y target 0.80. `info["truck_departures"]` es una lista con `truck`, `departure_t`, `soc_departure`, `target_soc`, `energy_deficit_kwh`, `departure_penalty_usd` y `departure_violation`; es vacía si no hay salida. `truck_departure_soc_gaps` conserva el déficit SOC devuelto por advance_interval para contrastarlo con el cálculo económico.

Salidas programadas: truck5 en horas 8, 32, 56, 80, 104 y 128; truck6 en 32 y 102. Son ocho eventos: lunes a sábado 08:00 para truck5; martes 08:00 y viernes 06:00 para truck6. Dos salidas simultáneas se suman. El cierre en 168 no crea una salida ni penalización artificial.

Una no convergencia o excepción del solver no confirma SOC ni cobra la salida. Se registra el intento fallido y se permite reintentar la transición.

Ejemplo únicamente de prueba, con c=2 y capacidad=315: SOC=0.70 genera 31.5 kWh de déficit y USD 63; SOC=0.60 genera 63 kWh y USD 126; SOC=0.80 o 0.90 genera cero. El test integrado verifica cobro una sola vez y SOC sin elevarlo al target. Otro test comprueba explícitamente el SOC posterior a la acción.

## 8. Métricas episódicas disponibles

`env.get_episode_metrics()` devuelve una copia acumulada, también antes del final; consultarla no suma otra vez. `include_runtime=False` excluye únicamente tiempos de pared. Al terminar, `info["episode_metrics"]` contiene esa copia determinista y `info["episode_runtime"]` los tiempos. Reset reinicia la agregación.

| Grupo | Claves y definición |
| --- | --- |
| Retorno/costos | return_total; energy_cost_usd; pf_penalty_usd; truck_departure_penalty_usd: sumas horarias. |
| Energía de red | import_kwh=Σmax(Pgrid,0)Δt; export_kwh=Σmax(−Pgrid,0)Δt; losses_kwh=ΣPlossesΔt. |
| PV | pv_available_kwh; pv_utilized_kwh; curtailment_kwh; pv_by_plant con esos tres totales por planta; pv_utilization_fraction (None si no hay disponible). Disponible es potencia efectiva; utilizada es inyección confirmada; curtailment=max(disponible−utilizada,0). |
| Voltajes ABC | voltage_min_pu_phase y voltage_max_pu_phase: extremos de los buses LV bus2–bus6 en todo el episodio, orden A/B/C. |
| Conteos V | voltage_violation_steps: paso con cualquier nodo/fase fuera de [0.95,1.05]; voltage_violation_steps_phase por fase; voltage_violation_node_phase_count cuenta ocurrencias nodo/fase/hora. |
| Magnitud V | g=max(0,0.95−V)+max(0,V−1.05); voltage_violation_max_pu=max(g); voltage_violation_sum_pu=Σg sobre buses/fases/pasos; voltage_violation_integral_pu_h=ΣgΔt. |
| PF ABC | pf_min_phase, pf_max_phase, pf_mean_phase (media temporal aritmética), pf_last_phase; pf_violation_steps y pf_violation_steps_phase para PF<0.90; pf_violation_max y pf_violation_sum de max(0,0.90−PF). |
| BESS | bess_soc_initial, bess_soc_final, bess_energy_initial_kwh, bess_energy_final_kwh: inicial 0.50/125 kWh y último estado confirmado. |
| Trucks | truck_departure_count, truck_departure_violation_count, truck_energy_deficit_kwh, truck_departure_penalty_usd, truck_departures con todos los SOC/eventos. |
| Red | vuf_max_pct: máximo VUF de tensión LV; transformer_loading_max_pct: máximo del loading existente; balance_residual_max_abs_kw: máximo residual activo absoluto. |
| Convergencia | solver_attempts, converged_steps, failed_solver_attempts, all_solver_attempts_converged, last_solver_error. Cuentan intentos de transición, no cada llamada interna de OpenDSS. |
| Runtime | runtime.reset_seconds, step_seconds_total, step_seconds_max, elapsed_wall_seconds. Incluye intentos fallidos en total de step. El tiempo de pared incluye pausas externas desde reset y queda fijo al concluir. |
| Contexto | steps, complete, split, scenario_id, c_unserved_usd_per_kwh, lambda_pf, phase_order, voltage_buses, voltage_limits_pu. |

Los extremos/medias por fase son None mientras no exista un paso válido; acumulados y conteos empiezan en cero. Los snapshots son serializables a JSON. `all_solver_attempts_converged` se interpreta junto con steps y complete: no implica episodio terminado. Valores exactamente 0.95/1.05 y PF=0.90 no cuentan como violaciones.

El test BESS descarga hasta su límite histórico 0.08 / 20 kWh y finaliza allí, partiendo de 0.50 / 125 kWh. Verifica las 168 recompensas y la ausencia de salvage value, costo de SOC final, restricción terminal u obligación de recuperar energía inicial.

## 9. Reproducibilidad e interfaz

Formato de identidad canónica: `benchmark_v1:<12 caracteres SHA manifiesto>:<split>:<office_id>|<wh1_id>|<wh2_id>|<solar_id>`.

Con el mismo manifiesto, split, ID, seed y acciones se obtienen exactamente las mismas observaciones, recompensas, estados y telemetría física en el stack verificado, con extreme_events=None. Runtime queda fuera de esa identidad. No se afirma igualdad bit a bit entre plataformas o versiones de NumPy/OpenDSS no verificadas.

`reset(seed=...)` domina. Si el primer reset no recibe seed y el constructor sí, utiliza la del constructor. Para reproducir se registra siempre una seed explícita: reset(seed=None) posterior continúa el generador y por sí solo no identifica una realización.

La selección de perfiles tiene un stream independiente sin consumir PF ni reparto ABC. Cada sampler recibe el week_idx del split seleccionado. Se mantienen los procedimientos de ruido y forecasts. Una seed histórica puede producir otro episodio respecto al ambiente anterior por el cambio de pools/selección; no se pretende reproducir las antiguas semanas mal alineadas.

Ejemplo de uso futuro; `valor_cientifico_aprobado` debe proceder de una decisión explícita, sin búsqueda en validation/test:

```python
from env.Ambiente_industrial import MicrorredIndustrial
from env.scenario_catalog import ScenarioCatalog

env = MicrorredIndustrial(
    split="train",
    c_unserved=valor_cientifico_aprobado,
    lambda_pf=1.0,
    extreme_events=None,
)
obs, info = env.reset(seed=123)
scenario_id = info["scenario_id"]
obs, info = env.reset(seed=123, options={"scenario_id": scenario_id})
validation_ids = ScenarioCatalog().reference_scenario_ids("optuna_validation")
```

También puede fijarse scenario_id en el constructor. `MicrorredIndustrial(split="train")` funciona para inspección; reset informa que falta c_unserved. Se preservan 114 observaciones, 12 acciones y metadata exclusivamente en info, clipping existente y sentinel terminal de 114 ceros con terminated=True/truncated=False a las 168 horas.

Los forecasts siguen siendo **synthetic noisy forecasts**, realizaciones futuras perturbadas, con horizonte de 11 h y extensión cíclica de las primeras 11 horas de la misma realización semanal. Los tests verifican la identidad forecast=realización+ruido y que los samplers de train solo contienen perfiles train. No son predicciones entrenadas ni una demostración de causalidad para un sistema real.

## 10. Tests realizados y resultados

Resultado final: **140 pasados, 0 fallidos, 0 errores, 0 omitidos dentro de la selección nominal**. Son 18 pruebas nuevas y 122 existentes. Duración del ejecutor: 29.61 s; dato descriptivo, sin optimización de rendimiento.

| Módulo | Tests pasados |
| --- | --- |
| test_action_resolution | 9 |
| test_benchmark_v1 | 18 |
| test_electrical_instrumentation | 6 |
| test_electrical_snapshot_2e | 13 |
| test_environment_reset_2c | 11 |
| test_environment_step | 7 |
| test_episode_render | 2 |
| test_exogenous_realization_layers | 5 |
| test_load_telemetry | 5 |
| test_opendss_easy_1c_b | 12 |
| test_price_sampler | 8 |
| test_pv_action_decoder_2d | 9 |
| test_pv_spatial_variation_2d1 | 6 |
| test_reward | 10 |
| test_temporal_contract_2b | 9 |
| test_transformer_production | 5 |
| test_vehiculos_zoe | 1 |
| test_vminpu_constant_power | 4 |

Aceptación A–G: partición disjunta/exhaustiva; reconstrucción exacta de timestamps y valores; dos trayectorias completas idénticas con acciones aleatorias prefijadas; costo proporcional, SOC posterior a acción y salidas simultáneas; identidad del reward; BESS terminal intacto; regresiones de signos, capability, balances, flujo ABC, VUF y valores de referencia del transformador.

`gymnasium.utils.env_checker.check_env(..., skip_render_check=True)` pasa. Emite dos advertencias por límites infinitos existentes del Box de observación (cargas/forecasts); no se alteró el contrato para silenciarlas. La primera corrida encontró solo una comparación incorrecta del runtime entre episodios. Se corrigió para excluir tiempo de pared manteniendo todas las variables físicas; las corridas posteriores pasaron.

Los snapshots de regresión eléctrica incluyen cargas y acciones exigentes preexistentes, pero no activan extreme_events ni generan episodios de stress-testing. Se aplazan cuatro módulos `test_extreme*` y cinco casos mixtos que sí activan eventos. No se contabilizan como pasados; sus nombres figuran en benchmark_v1_checks.json. No se validó ahora la reproducibilidad de eventos.

Ejecución desde CODIGO con el Python de anmrl ya instalado:

```powershell
& "python" -B -X utf8 -B scripts/run_benchmark_v1_checks.py
```

Reconstrucción determinista de derivados, sin escribir sobre fuentes:

```powershell
& "python" -B -X utf8 -B scripts/prepare_benchmark_v1_data.py --verify-only
```

| Runtime/dependencia | Versión verificada |
| --- | --- |
| Python | 3.11.14 |
| numpy | 1.26.4 |
| pandas | 1.5.3 |
| gymnasium | 1.2.3 |
| opendssdirect.py | 0.9.4 |
| dss-python | 0.15.7 |
| dss-python-backend | 0.14.5 |

No se instalaron dependencias. La comprobación adicional detectó cero perfiles efectivos idénticos entre splits después del escalado individual y del floor solar existente. La reconstrucción completa conserva los trece hashes. `git diff --check` terminó limpio.

## 11. Limitaciones que permanecen

- **BESS terminal libre:** un agente puede aprovechar el cierre del horizonte consumiendo energía inicial. Se registra, sin corrección económica.
- **Neutro:** Yy0 con Rneut=−1 y buses cuyo cuarto conductor referencia nodo 0. El test verifica NodeOrder `[1,2,3,0,1,2,3,0]` y 18 nodos totales. No existe conductor neutro distribuido. Se conservan topología, parámetros, cargas, signos y Solve/CalcVoltageBases.
- **Calendario:** solar sin fechas verificadas; train combina distintas fechas y edificios; el split no es estrictamente prospectivo. Seis escenarios base de referencia por conjunto de evaluación siguen siendo seis bases, aunque se repitan con muchas seeds.
- **Datos/economía:** se conserva el escalado semanal por pico, el perfil de precios exógeno y la tarifa común de compra/venta. Se filtran faltantes sin imputación. lambda_pf=1.0 es la configuración existente, no una calibración regulatoria nueva.
- **Forecasts:** se conserva el supuesto sintético construido desde el futuro realizado y su extensión semanal.
- **Eventos:** extreme_events=None en los estudios iniciales. Su reproducibilidad con semillas ausentes requiere trabajo posterior antes de stress-testing.
- **Separación operativa:** el ambiente solo carga su split y rechaza su cambio público, pero los datos no están cifrados ni inaccesibles al propietario del repositorio. El futuro controlador Optuna deberá usar exclusivamente optuna_validation; aún no existe un controlador cuyo acceso pueda auditarse. final_test se reserva hasta congelar hiperparámetros y entrenar las configuraciones finales.
- **Alcance:** las pruebas demuestran funcionamiento y regresión; no estabilidad del aprendizaje, rendimiento SAC ni seguridad por restricciones. No se implementan Lagrangian SAC, conformal prediction ni safety projection.

## 12. Decisiones pendientes antes de entrenar

La única decisión científica pendiente introducida por estos cambios es **fijar un único c_unserved positivo, con justificación independiente de validation/test**. Debe permanecer idéntico en baseline SAC y futuras comparaciones, registrado junto al lambda_pf conservado, hashes, seeds y protocolo. No se seleccionará mediante Optuna.

El ambiente está técnicamente preparado para smoke training una vez fijado ese coeficiente. El entrenamiento está fuera de esta fase y no se ha realizado. Antes de ejecutar Optuna corresponde implementar/configurar el controlador de entrenamiento y evaluación, completar el smoke training autorizado y fijar el protocolo sobre las seis composiciones de optuna_validation, con seeds registradas. final_test queda reservado. Son actividades futuras de entrenamiento, no fallos abiertos del ambiente.

No se requiere rediseñar el neutro, cambiar la condición terminal BESS, retirar PF ni modificar forecasts para habilitar el smoke training. Las decisiones del alcance actual ya se ejecutaron.



## 13. Portable private snapshot addendum (2026-09-09)

This addendum documents packaging only. The historical audit remains subordinate to this changelog. Original source files were not edited. Current own-code defaults are `c_unserved=1.0` USD/kWh and `lambda_pf=1.0`, both configurable; they were already present in the current environment and campaign source and are explicitly required for this snapshot. Sections 6, 9 and 12 above retain older statements that the coefficient is unset: those statements do not describe the supplied current code. No reward coefficient or physical logic was changed during packaging. The source runner/tests label 1.0 as the Phase 2 approved default; the historical narrative had not been updated.

- Environment modules are packaged in `env/`; utilities in `scripts/`; campaign infrastructure in `optuna/`. The latter is a CLI directory, deliberately without `__init__.py`, to avoid shadowing the installed Optuna package.
- Imports and file paths are portable; default manifest: `data/scenario_splits.json`. Historical references to unshipped files are labeled rather than recreated.
- Only train and optuna_validation NPZ files are present. All eight are byte-identical to the current derivatives. Holdout metadata remains, but no physical final_test file is included or opened.
- Original manifest hash: `0db4a1ccbceec4833d6c94a131efe9e5060b6e09b43198522563269ff2c0f542`.
- Snapshot manifest hash: `91b28e5d82d79da488f167eb7cd3d9dcb9ee805b3726dc232b63535e62c91830`. Exactly twelve NPZ path strings changed; profile/file hashes and other metadata are unchanged. The canonical scenario-ID prefix consequently changes. The campaign pins the new manifest hash.
- The provided 17 requested test files are included. `test_episode_render.py` is additionally retained because its two nominal checks belong to the approved 140-test selection. No listed source file was missing. Minimal `env/__init__.py` and `scripts/__init__.py` files define the packages.
- Five previously deferred event methods were removed from the mixed telemetry/render test files. The four separate extreme-event suites and the SAC training/controller test suite are excluded. No deferred case is labeled as passed.
- Two raw/holdout-dependent acceptance methods now have accurate derivative-level names: `test_included_dated_profiles_reconstruct_manifest_timestamps_and_hashes` and `test_included_derived_hashes_solar_metadata_and_absent_holdout`. They test distributed arrays against frozen profile hashes/calendar metadata, not full original-source reconstruction. The other 138 nominal cases retain their assertions, with import/path adaptation and train-only sampler fixtures where needed.
- `prepare_benchmark_v1_data.py --verify-only` validates the snapshot. Rebuilding requires explicit original inputs and a fresh output directory; the copied builder never exports physical final_test data.
- SAC snapshots, data-access auditing, VM worker paths and campaign code fingerprints follow the new folders. Campaign science and budgets are preserved: 60 trials, 12 random warm-up, 48 TPE, at most 100800 steps/trial, separate processes, final_test forbidden. A sanitized workstation-name placeholder fails closed until configured for future deployment.
- Original software code is licensed under PolyForm Noncommercial License 1.0.0; original documentation prose is separately licensed under CC BY-NC-SA 4.0. Neither license automatically applies to third-party or derived datasets. See `DATA_PROVENANCE.md` for source-specific terms and open solar/parameter attribution.

No training, Optuna, remote repository creation or push was performed for this snapshot. Results and local delivery location are recorded in the separate `GITHUB_REPO_PREP_REPORT.md`; it is outside the upload folder to keep personal paths out of the repository.


## 14. License-only update (2026-09-09)

The default license for all original software code is PolyForm Noncommercial License 1.0.0. The complete official text is reproduced unchanged in LICENSE; README.md identifies the copyright holders and commercial-licensing contact policy. The software is source-available for non-commercial research and educational use.

Original documentation prose is separately licensed under CC BY-NC-SA 4.0, as specified in README.md. Original code examples remain under the software license. Third-party materials, quoted license texts and notices retain their own terms. Third-party and derived datasets remain subject to the terms, licenses and attribution requirements of their respective original sources; neither project license automatically covers them.

Only LICENSE, README.md, DATA_PROVENANCE.md and this changelog changed within the repository. The separate preparation report was updated for consistency. All other repository files, including software, data, tests, manifests, configuration and dependencies, were verified byte-identical by SHA-256. No benchmark execution, training or Optuna was performed for this license-only update.
