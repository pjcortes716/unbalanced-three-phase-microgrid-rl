> Snapshot portable: 60 trials, 12 warm-up + 48 TPE, max 100800 steps/trial; final_test forbidden; multiprocessing, not threads. No campaign was started during this preparation. Before any later launch, set `blocked_local_hostname` in `optuna/vm_campaign.json`; the distributable placeholder deliberately prevents launch until configured.

# Campaña benchmark_v1 SAC / Optuna en la VM

Estado al preparar este paquete: código y pruebas de protocolo preparados en el PC; **preflight concurrente y optimización pendientes de ejecución en la VM**. No se dispone aún de host/IP, usuario, sistema operativo ni autenticación SSH. Las comprobaciones locales no acreditan rendimiento, concurrencia real ni compatibilidad de las dependencias en la VM.

## Presupuesto y ejecución

El lanzador es vm_campaign.py y la configuración congelada es vm_campaign.json.

| Etapa | Trials | Procesos máximos | Pasos por trial | Poda |
|---|---:|---:|---:|---|
| Preflight técnico, estudio separado | 4 | 4 | 2000 | Ninguna |
| A: inicialización aleatoria | 12 | 6 | 100800 | Ninguna |
| B: TPE paralelo | 48 | 28 | Hasta 100800 | Desde 50400 |

Los cuatro trials técnicos no forman parte de las 60 observaciones de optimización. No se reemplazan automáticamente trials fallidos con trials adicionales: completos + podados + fallidos nunca exceden 60. La etapa B espera que los 12 primeros hayan terminado. Si ninguno completa por fallos técnicos, se detiene la campaña para investigar.

Se reserva capacidad para SO/coordinación manteniendo un máximo de 28 procesos de trabajo en una VM de al menos 30 CPU disponibles. El coordinador comprueba hostname y recursos antes de importar Optuna; el PC local definido en `blocked_local_hostname` está bloqueado explícitamente.

Cada trabajo se ejecuta con un nuevo proceso Python, su propio ambiente, OpenDSS, SAC, replay buffer y directorio. No se usa study.optimize con n_jobs ni se comparte una instancia de OpenDSS entre procesos. El directorio actual, DataPath de OpenDSS y las cachés del worker apuntan a su trial. Solo el estudio persistente se comparte para reportar resultados y coordinar la barrera técnica.

Antes de importar NumPy/Torch/OpenDSS/SB3 se fijan OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS y NUMEXPR_NUM_THREADS a 1. Torch usa un hilo intraop y uno interop; threadpoolctl comprueba también los pools BLAS/OpenMP.

## Invariantes científicos y espacio de búsqueda

Ambiente sin modificaciones adicionales: c_unserved=1.0, lambda_pf=1.0, extreme_events=None, episodios de 168 horas, observación de 114 dimensiones y acción de 12. Se conserva el reward y todos los límites eléctricos. Entrenamiento solo en train; evaluación solo en optuna_validation.

| Parámetro | Valores |
|---|---|
| learning_rate | Logarítmico [0.00005, 0.001] |
| net_arch | [128,128], [256,256], [256,256,256] |
| batch_size | 128, 256, 512 |
| gamma | 0.99, 0.995, 0.999 |
| tau | Logarítmico [0.002, 0.02] |
| learning_starts | 1680, 3360, 8400 |
| ent_coef | auto_0.01, auto_0.1, auto_1.0 |

Fijos: buffer_size=100000, train_freq=1, gradient_steps=1, target_entropy="auto" de SB3, MlpPolicy, CPU. La seed de aprendizaje es 42 para todos los trials; cada episodio de train conserva la derivación de semillas del controlador ya aprobado.

Los primeros 12 usan RandomSampler. El coordinador asigna los parámetros de TPE a partir del trial 12 usando el mismo estudio, n_startup_trials=0 y constant_liar=True. Las observaciones iniciales completas ya están disponibles al empezar B. La semilla del sampler es 20260908 + número de trial; se conserva cada combinación asignada. El orden de terminación asíncrono puede cambiar propuestas posteriores, por lo que no se promete repetir bit a bit una campaña paralela completa. [TPESampler oficial](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

## Validación y objective

Checkpoints exactos: 16800, 33600, 50400, 67200, 84000 y 100800 pasos de interacción con train. Los pasos de evaluación no incrementan el presupuesto de entrenamiento.

Se utilizan los seis IDs devueltos, en orden, por reference_scenario_ids("optuna_validation"). Para la semana i=0..5 se fijan las semillas 420001+i y 430001+i. Son doce realizaciones por checkpoint, idénticas en todos los trials, con deterministic=True. El protocolo expandido con IDs y seeds queda en validation_protocol.json de cada trial. Los estados RNG de Python, NumPy y Torch se restauran después de evaluar para que la validación no consuma aleatoriedad del entrenamiento.

El objective es exclusivamente la media de return_total de esas doce trayectorias. Un trial completo entrega la media en 100800; uno podado conserva el último checkpoint e intermediate value. Se registran por separado energy_cost_usd, pf_penalty_usd, truck_departure_penalty_usd, truck_energy_deficit_kwh, voltage_violation_steps/rate/max_pu, pf_violation_steps/rate, losses_kwh, pv_utilization_fraction, bess_soc_final, vuf_max_pct y transformer_loading_max_pct. Las métricas por episodio conservan además el detalle disponible. No se utilizan como filtros de selección.

MedianPruner: n_startup_trials=5, n_warmup_steps=50400, interval_steps=16800, n_min_trials=5. Una segunda condición explícita impide podar los primeros 12 trials y evita etiquetar como podado un trial que llegó a 100800. Se eligió de antemano el umbral conservador de 50400; no se infiere aquí ninguna varianza observada. [MedianPruner oficial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html).

## Storage y reanudación

La configuración inicial usa JournalStorage con JournalFileBackend y JournalFileOpenLock en el disco local de la VM. Se rechazan mounts de red habituales: NFS, CIFS/SMB y SSHFS. No se usa SQLite. [Paralelización oficial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html), [JournalFileBackend](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalFileBackend.html).

Si PostgreSQL ya está disponible fácilmente, se puede cambiar storage a postgresql **antes de iniciar el preflight** y configurar OPTUNA_STORAGE_URL en el entorno de la VM. No se incluye esa URL en el paquete ni en config.json. La VM requiere el driver psycopg; instalarlo en su entorno Python y usar una URL postgresql+psycopg://. El adaptador RDB limita el pool por proceso; el coordinador mantiene además un advisory lock PostgreSQL por nombre de estudio.

Un único coordinador asigna trials mediante ask, fija parámetros y entrega a cada worker el ID persistente; los workers reportan y terminan ese trial mediante tell. Optuna está fijado en 5.0.0 porque el traspaso usa el constructor Trial(study, trial_id) y el identificador interno del trial. El coordinador conserva PID y fecha de creación de cada proceso y reconcilia estados tras reinicios. No se atribuye esta recuperación al heartbeat de RDBStorage: su heartbeat automático no cubre el flujo ask/tell. [RDBStorage oficial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html).

Para reanudar se utiliza **el mismo comando, directorio, VM y configuración**, sin borrar archivos. El estudio queda vinculado a un campaign_id; se rechazan archivos fuente, manifiesto, Python, paquetes instalados o parámetros de configuración distintos. Los workers vivos continúan y el coordinador los adopta. Los workers muertos quedan como FAIL, conservan checkpoints y cuentan dentro del máximo de 60.

La reanudación implementada es de la campaña: **no restaura a mitad de un trial el replay buffer, optimizadores y RNG desde el último checkpoint**. Un checkpoint SAC permite cargar la política y preservar el resultado parcial; no se declara equivalente a continuar el mismo aprendizaje. No se relanzan trials muertos de forma silenciosa.

## Preflight exclusivo en la VM

El único entrenamiento técnico previo son cuatro procesos de 2000 pasos, learning_starts=500, seed=42, arquitectura y SAC del smoke reducido anterior. No se repite el smoke completo ni la suite de 140 tests.

Cada proceso registra un nombre de circuito exclusivo por PID en el storage y espera una barrera de los cuatro procesos antes de comprobar que su circuito sigue siendo propio. Después verifica replay_buffer_size=2000, 1500 actualizaciones, once episodios completos y el parcial restante.

Este preflight evalúa solo una realización de validation después de entrenar y la repite tras guardar/destruir/cargar el modelo. Se comprueba reproducción de trayectorias con tolerancia 1e-7 y coincidencia entre los cuatro hashes de trayectorias con la misma seed. Esto no evalúa calidad de la política.

El coordinador exige: cuatro procesos concurrentes y distintos, barrera de storage, circuitos distintos, límites de hilos, métricas persistidas con el número esperado de filas, save/load correcto, ausencia de NaN/Inf y fallos OpenDSS, y cero accesos o intentos bloqueados a datos prohibidos. resources.csv registra CPU/RAM aproximadamente cada cinco segundos.

Si pasa, se crea preflight_passed.json y se inicia automáticamente la campaña de 60. Si falla, se escribe preflight_failed.json y no se inicia optimización. Un relanzamiento conserva ese resultado y no ejecuta otros cuatro entrenamientos de manera automática.

## Protección de final_test

El ZIP contiene solo ocho NPZ: cuatro fuentes de train y cuatro de optuna_validation. No incluye los NPZ de final_test, los pools originales ni resultados/modelos anteriores. El manifiesto data/scenario_splits.json conserva toda la metadata y los hashes de datos; sus rutas portables producen este nuevo SHA-256:

    91b28e5d82d79da488f167eb7cd3d9dcb9ee805b3726dc232b63535e62c91830

La factoría del controlador bloquea final_test antes de crear el ambiente. DataAccessAudit registra cada apertura de NPZ y bloquea splits incorrectos antes de abrir el archivo. Leer los nombres y hashes del manifiesto no equivale a leer series temporales de final_test. La entrega final exige final_test accesses=0; también se informan los intentos bloqueados.

## Instalación y arranque pendientes en la VM

Requiere Python 3.11. Las versiones base reproducen el stack del smoke local aprobado; Optuna, psutil y threadpoolctl se agregan solo en la VM. Se registra pip_freeze.txt de la VM.

Ejemplo para una VM Linux, después de transferir y extraer el paquete en una carpeta nueva de disco local. Ajustar las rutas al acceso que se proporcione:

    python3.11 -m venv .venv
    .venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/python -m pip install -r optuna/requirements_vm.txt
    .venv/bin/python -m pip check

El índice CPU es el publicado para Torch 2.10.0 por PyTorch. No se instalan torchvision/torchaudio, que este pipeline no requiere. [Instalación oficial](https://pytorch.org/get-started/previous-versions/).

En una VM Windows, usar el Python 3.11 disponible para crear .venv y .venv\Scripts\python.exe en lugar de .venv/bin/python. El lanzador admite ambos sistemas y crea procesos ocultos en Windows.

Arranque único desde un terminal persistente de la VM; VM_HOSTNAME debe coincidir con socket.gethostname() allí:

    .venv/bin/python -B optuna/vm_campaign.py --config optuna/vm_campaign.json --output runs/benchmark_v1_sac_vm60 --vm-hostname VM_HOSTNAME

El comando contiene tanto el preflight como la campaña. No se debe lanzar un segundo coordinador. El bloqueo local y, si corresponde, el advisory lock PostgreSQL impiden asignaciones simultáneas. El PC local no está autorizado para ejecutar ese comando con entrenamiento.

La conectividad SSH, SO, disponibilidad de Python 3.11, espacio de disco, dependencias y PostgreSQL todavía deben comprobarse en la VM. No se han ejecutado los comandos de instalación ni el lanzador durante la preparación local.

## Artefactos y entrega

Cada trial_XXXX contiene job.json, params.json, config.json, environment_manifest.json, validation_protocol.json, process.json, status.json, worker.log, training_log.csv, episode_metrics.csv, validation_episodes.csv, validation_history.csv, validation_reloaded.csv, episode_events.jsonl y data_access.jsonl. Guarda un ZIP por checkpoint alcanzado y model_final.zip para el final o la poda. Seed, timesteps, episodios, actualizaciones, wall time, recursos y comprobación de save/load se conservan en status/config.

El directorio global contiene el estudio persistente, campaign_metadata.json, pip_freeze.txt, resources.csv, preflight_passed/failed.json y, al terminar, global_summary.json, top_10.csv y OPTUNA_VM_RESULTS.md.

El resumen final informa completos/podados/fallidos, wall time desde el primer inicio (incluye preflight e interrupciones), máximos muestreados CPU/RAM, top diez completos por objective final, parámetros mejores, return por checkpoint, cambio inicial-final y rango de los últimos tres checkpoints. Estos últimos son descriptores de evolución y no una prueba multisemilla.

Se proponen hasta cinco configuraciones completas distintas por return final para la confirmación futura; si por fallos no hay al menos tres, se explicita. Los procesos muertos sin cierre limpio se identifican como telemetría incompleta; no se inventa su consumo ni se asegura ausencia de incidentes no observados.

Al terminar los 60 trials, el programa finaliza. Repetir el comando después de generar el resumen solo informa que la campaña terminó. No se ejecuta confirmation, Lagrangian SAC ni ningún entrenamiento posterior.

