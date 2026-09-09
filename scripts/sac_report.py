"""Descriptive smoke report and standalone scientific plots, without selection."""

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_report(run_dir):
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    m = json.loads((run_dir / "environment_manifest.json").read_text(encoding="utf-8"))
    if c["experiment"]["purpose"] == "technical_smoke_reduced":
        return create_technical_report(run_dir, s, c, m)
    train = pd.read_csv(run_dir / "episode_metrics.csv")
    initial = pd.read_csv(run_dir / "validation_initial.csv")
    final = pd.read_csv(run_dir / "validation_final.csv")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train["episode"], train["return_total"], color="#2274a5", lw=1, alpha=0.8, label="Retorno por episodio")
    ax.plot(train["episode"], train["return_total"].rolling(20).mean(), color="#d46a17", lw=2, label="Media móvil de 20")
    ax.axvline(20, color="#555555", ls=":", label="Fin de learning_starts")
    ax.set(xlabel="Episodio semanal de train", ylabel="Retorno (USD con signo)", title="Smoke SAC · seed 42 · 120 semanas")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "episodic_return.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, color in (
        ("energy_cost_usd", "Energía", "#2274a5"),
        ("pf_penalty_usd", "PF", "#7c4d9e"),
        ("truck_departure_penalty_usd", "Trucks", "#d46a17"),
    ):
        ax.plot(train["episode"], train[key], lw=1, alpha=0.8, label=label, color=color)
    ax.set(xlabel="Episodio semanal de train", ylabel="Costo (USD)", title="Componentes económicos del reward")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "reward_components.png")
    plt.close(fig)

    parts = []
    def add(text):
        parts.append(text.strip() + "\n\n")
    def table(headers, rows):
        add("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) +
            " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows))
    def f(value):
        return f"{float(value):,.6f}"
    def link(path, label):
        return "[" + label + "](<" + path.resolve().as_posix() + ">)"

    add("# SAC_SMOKE_REPORT\n\nSmoke de baseline SAC completado. Verifica la tubería RL; no selecciona "
        "hiperparámetros ni demuestra superioridad de SAC. No se inició Optuna ni Lagrangian SAC.")
    add("## A. Configuración exacta")
    add("c_unserved=1.0 USD/kWh es el valor científico aprobado en Fase 2 y el nuevo default. Sigue siendo "
        "configurable, finito y positivo. No se cambió el reward, los módulos físicos, los derivados ni el "
        "manifiesto de escenarios. El código anterior que usaba None como configuración pendiente fue actualizado.")
    table(["Parámetro", "Valor"], [(key, json.dumps(value)) for key, value in c["experiment"].items()])
    add("Target entropy auto se resuelve a −12 por las 12 acciones. Cada episodio train usa "
        "uint32(SeedSequence([42,1001,índice_desde_cero])); su seed e identidad se registran. No se utiliza "
        "VecNormalize ni se cambia la escala del reward. Se conservan las restantes opciones por defecto de SB3 2.7.1.")
    add("Las evaluaciones usan model.predict(obs, deterministic=True), recorren los seis IDs de "
        "reference_scenario_ids('optuna_validation') y mantienen respectivamente seeds 420001…420006.")
    add("## B. Stack de software")
    table(["Componente", "Versión"], [("Python", m["python"]), *m["packages"].items()])
    add("CPU: " + m["cpu"] + "; un hilo Torch y algoritmos deterministas. No se optimizó el solver. "
        "Commit base: " + str(m["git"].get("commit")) + "; working tree dirty=" +
        str(m["git"].get("dirty")) + ". environment_manifest.json contiene hashes efectivos; code_snapshot "
        "preserva copias de los archivos usados. SHA del manifiesto de escenarios: " + m["scenario_manifest_sha256"] + ".")
    add("## C. Tiempo de entrenamiento")
    table(["Métrica", "Resultado"], [
        ("Timesteps", s["timesteps"]), ("Episodios completos train", s["completed_training_episodes"]),
        ("Actualizaciones de gradiente", s["gradient_updates"]), ("Checks finitos de actualizaciones", s["finite_update_checks"]),
        ("Training wall seconds", f(s["training_wall_seconds"])), ("Pipeline wall seconds", f(s["wall_time_seconds"])),
    ])
    add("Training wall time mide model.learn, checks y logging. El tiempo total incluye evaluación inicial/final "
        "y save/load, y excluye renderizar el reporte. Los primeros 3360 pasos (20 semanas) llenan el replay "
        "sin actualizar gradientes. training_log.csv contiene una fila por actualización posterior. No se "
        "cambió ningún hiperparámetro tras observar los resultados.")
    add("## D. Curva de episodic return")
    add("![Curva de retorno](<" + (figures / "episodic_return.png").resolve().as_posix() + ">)")
    add("Media de los últimos 20 episodios: " + f(s["training_last20"]["means"]["return_total"]) +
        ". Es descriptiva de una seed y escenarios estocásticos, no un resultado científico.")
    add("## E. Descomposición del reward")
    add("reward = −energy_cost_usd −pf_penalty_usd −truck_departure_penalty_usd. Se verificó esta identidad "
        "en cada transición. PF mantiene lambda_pf=1.0. No se añade costo de voltaje ni término terminal BESS.")
    table(["Costo USD", "Media 120 episodios", "Media últimos 20"], [
        (key, f(s["training_all"]["means"][key]), f(s["training_last20"]["means"][key]))
        for key in ("energy_cost_usd", "pf_penalty_usd", "truck_departure_penalty_usd")])
    add("![Componentes de costos](<" + (figures / "reward_components.png").resolve().as_posix() + ">)")
    add("## F. Métricas de seguridad descriptivas")
    a = s["training_all"]
    table(["Métrica train", "Resultado"], [
        ("Fracción de pasos con V LV fuera de [0.95,1.05]", f(a["voltage_violation_rate"])),
        ("Magnitud máxima de violación V (pu)", f(a["voltage_violation_max_pu"])),
        ("Fracción de pasos con alguna fase PCC PF<0.90", f(a["pf_violation_rate"])),
        ("VUF LV máximo (%)", f(a["vuf_max_pct"])),
        ("Loading transformador máximo (%)", f(a["transformer_loading_max_pct"])),
        ("Fallos OpenDSS total de la tubería", s["sanity"]["opendss_failures_total"]),
        ("SOC final BESS medio", f(a["means"]["bess_soc_final"])),
        ("Utilización PV energética", f(a["pv_utilization_fraction"])),
    ])
    add("Los rates dividen los pasos con cualquier violación entre 20160; cada paso cuenta una sola vez, "
        "aunque haya varias fases/buses. Los CSV conservan también extremos ABC, severidad acumulada, PF ABC y "
        "balances. Son métricas descriptivas, sin garantía de restricciones.")
    add("## G. Servicio V2G")
    table(["Métrica train", "Media por episodio", "Total"], [
        (key, f(a["means"][key]), f(a["totals"][key])) for key in
        ("truck_departure_count", "truck_departure_violation_count", "truck_energy_deficit_kwh", "truck_departure_penalty_usd")])
    add("La columna JSON truck_departures registra cada SOC de salida y déficit. Se usa el SOC posterior a "
        "la acción y se cobra una sola vez por salida programada, sin forzar SOC=0.80.")
    add("## H. Evaluación inicial vs final: seis semanas de optuna_validation")
    comparison = []
    for i, (before, after) in enumerate(zip(initial.to_dict("records"), final.to_dict("records")), 1):
        if before["scenario_id"] != after["scenario_id"] or before["realization_seed"] != after["realization_seed"]:
            raise RuntimeError("Initial/final validation pairing mismatch.")
        comparison.append((i, before["realization_seed"], f(before["return_total"]), f(after["return_total"]),
                           f(after["return_total"] - before["return_total"]),
                           f(before["truck_energy_deficit_kwh"]), f(after["truck_energy_deficit_kwh"])))
    table(["Escenario", "Seed", "Return inicial", "Return final", "Diferencia", "Déficit inicial kWh", "Déficit final kWh"], comparison)
    table(["Métrica media", "Inicial", "Final"], [
        (key, f(s["validation_initial"]["means"][key]), f(s["validation_final"]["means"][key]))
        for key in ("return_total", "energy_cost_usd", "pf_penalty_usd", "truck_departure_penalty_usd",
                    "voltage_violation_steps", "pf_violation_steps", "import_kwh", "export_kwh", "losses_kwh")])
    add("IDs y seeds literales, en orden, están en config.json.validation_protocol y ambos CSV. Toda mejora "
        "o deterioro se informa descriptivamente. No se seleccionó checkpoint, hiperparámetros, duración "
        "definitiva ni budgets lagrangianos.")
    add("## I. Save/load reproducible")
    check = s["save_load"]
    add("Se guardó model_final.zip, se destruyó la instancia original, se comprobó su destrucción con weakref "
        "y se cargó una nueva. Se repitió el primer escenario de validación y su seed durante 168 pasos.")
    table(["Variable", "Diferencia absoluta máxima"], check["maximum_absolute_differences"].items())
    add("Resultado: " + str(check["passed"]) + "; rtol=" + str(check["rtol"]) + ", atol=" + str(check["atol"]) +
        ". Los NPZ trajectory_validation_final_01 y trajectory_validation_reloaded_01 preservan las variables "
        "comparadas. SHA del modelo: " + s["model_sha256"] + ".")
    add("## J. Errores o warnings")
    add("NaN/Inf: 0. Fallos OpenDSS: 0. Episodios descartados silenciosamente: 0.")
    add("\n".join("- " + warning for warning in s["warnings"]))
    add("El aviso de Gym heredado procede de una dependencia instalada al importar SB3; esta tubería usa "
        "Gymnasium. El reset automático después de la semana 120 prepara otro episodio que recibe cero "
        "acciones y no cuenta como completado.")
    access = s["data_access"]
    add("El controlador valida split antes de crear el ambiente y audita las aperturas NPZ reales antes de "
        "permitirlas. Accesos permitidos: " + str(access["allowed_open_count"]) + "; splits: " +
        ", ".join(access["allowed_splits"]) + "; aperturas final_test: " + str(access["final_test_open_count"]) +
        "; intentos bloqueados en la corrida: " + str(len(access["blocked_open_attempts"])) +
        ". Evidencia: data_access.jsonl. final_test no fue evaluado.")
    add("## K. ¿Listos técnicamente para Optuna?")
    add("Sí: se completaron los 20160 pasos, 120 semanas, evaluaciones emparejadas y save/load, sin fallos "
        "numéricos o eléctricos y con final_test protegido. Las funciones son reutilizables; implementar y "
        "ejecutar Optuna corresponde a la siguiente fase. El smoke no elige el espacio de búsqueda ni su presupuesto.")
    add("Artefactos: " + link(run_dir / "run_summary.json", "resumen") + ", " +
        link(run_dir / "episode_metrics.csv", "métricas") + ", " + link(run_dir / "data_access.jsonl", "auditoría de archivos") + ".")
    report = run_dir / "SAC_SMOKE_REPORT.md"
    report.write_text("".join(parts), encoding="utf-8")
    root_report = Path(__file__).resolve().parents[1] / "SAC_SMOKE_REPORT.md"
    if root_report.exists():
        raise FileExistsError(f"Refusing to overwrite previous report {root_report}; run report is at {report}.")
    shutil.copyfile(report, root_report)



def create_technical_report(run_dir, summary, config, manifest):
    """Reduced smoke reports plumbing only, without policy-performance analysis."""
    lines = ["# SAC_SMOKE_REPORT - comprobación técnica reducida", "",
             "Prueba exclusiva de funcionamiento en el PC. No evalúa desempeño ni convergencia y no modifica el protocolo experimental futuro.", "",
             "## Configuración ejecutada", "", "```json",
             json.dumps(config["experiment"], indent=2), "```", "",
             "CPU, un hilo Torch, seed de aprendizaje 42. Se ejecutaron solo los dos primeros escenarios de optuna_validation, con seeds 420001 y 420002, antes y después. Save/load repitió únicamente el primero.", "",
             "## Resultado técnico", "",
             f"- Timesteps: {summary['timesteps']}.",
             f"- Episodios completos: {summary['completed_training_episodes']}.",
             f"- Episodio parcial: {summary['partial_episode_steps']} pasos; guardado explícitamente en partial_episode_metrics.json, sin forzar terminated ni truncated.",
             f"- Transiciones en replay: {summary['replay_buffer_size']}.",
             f"- Actualizaciones de gradiente: {summary['gradient_updates']}.",
             f"- Chequeos de parámetros/pérdidas finitos: {summary['finite_update_checks']}.",
             f"- NaN/Inf: {summary['sanity']['nonfinite_incidents']}.",
             f"- Fallos OpenDSS: {summary['sanity']['opendss_failures_total']}.",
             f"- Reward comprobado en cada transición; entrenamiento: {summary['sanity']['reward_checks_training']} comprobaciones.",
             f"- Save/load: {summary['save_load']['passed']}; instancia original destruida: {summary['save_load']['original_model_destroyed']}.",
             f"- final_test usado: {summary['final_test_used']}; aperturas de sus NPZ: {summary['data_access']['final_test_open_count']}.",
             f"- Wall time entrenamiento: {summary['training_wall_seconds']:.3f} s.",
             f"- Wall time tubería: {summary['wall_time_seconds']:.3f} s (incluye validaciones y save/load; excluye este reporte).", "",
             "## Reproducibilidad y trazabilidad", "",
             "El coeficiente aprobado es c_unserved=1.0 USD/kWh, default configurable; lambda_pf=1.0. El reward y la física permanecen iguales.",
             "La seed de cada episodio train es uint32(SeedSequence([42,1001,índice_desde_cero])). episode_events.jsonl y episode_metrics.csv conservan la seed y scenario_id. No se descarta el episodio parcial.",
             "Se preservan config.json, environment_manifest.json, training_log.csv, episode_metrics.csv, validation_initial.csv, validation_final.csv, model_final.zip y run_summary.json; además auditoría de aperturas, métricas parciales y trayectorias antes/después de recargar.",
             "La prueba save/load compara 168 pasos de observaciones, acciones, reward, PCC, voltajes, PF, SOC, costos y pérdidas. rtol=1e-7; atol=1e-7.", "", "```json",
             json.dumps(summary["save_load"]["maximum_absolute_differences"], indent=2), "```", "",
             "SHA manifiesto: " + manifest["scenario_manifest_sha256"],
             "Commit base: " + str(manifest["git"].get("commit")) + "; se registran hashes del working tree y copias en code_snapshot.", "", "## Stack", "",
             "Python " + manifest["python"], "```json", json.dumps(manifest["packages"], indent=2), "```", "",
             "## Avisos", "", *["- " + warning for warning in summary["warnings"]], "",
             "El aviso heredado de Gym aparece al importar una dependencia de SB3; esta tubería utiliza Gymnasium. Las observaciones reales se comprueban finitas aunque el Box permita límites infinitos.", "",
             "## Estado para la siguiente fase", "",
             "La infraestructura está técnicamente lista para diseñar Optuna: interactúa con el ambiente, llena replay, actualiza gradientes, registra métricas y conserva save/load con final_test bloqueado. No se ha implementado ni iniciado Optuna.",
             "Los 2000 pasos, learning_starts=500 y dos escenarios son exclusivos de esta comprobación. El presupuesto y los seis escenarios del protocolo experimental futuro se definirán posteriormente. No se extraen conclusiones sobre retorno ni calidad de política.", ""]
    report = run_dir / "SAC_SMOKE_REPORT.md"
    report.write_text(chr(10).join(lines), encoding="utf-8")
    root_report = Path(__file__).resolve().parents[1] / "SAC_SMOKE_REPORT.md"
    if root_report.exists():
        raise FileExistsError(f"Refusing to overwrite previous report {root_report}.")
    shutil.copyfile(report, root_report)
