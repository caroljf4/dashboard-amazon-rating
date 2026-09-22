"""Pruebas basicas de integridad de los datos y resultados del proyecto.

Ejecutar con: pytest -q
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def test_archivos_existen():
    for nombre in [
        "dataset_limpio.csv",
        "predicciones_prueba.csv",
        "comparacion_modelos.csv",
        "r2_regresion_lineal.csv",
        "importancia_variables.csv",
        "resumen_modelo.json",
    ]:
        assert (OUT / nombre).exists(), f"Falta el archivo {nombre}"


def test_dataset_limpio_sin_faltantes_clave():
    df = pd.read_csv(OUT / "dataset_limpio.csv")
    columnas_clave = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
    assert df[columnas_clave].isna().sum().sum() == 0
    assert len(df) > 0
    assert not df["id"].duplicated().any()


def test_predicciones_rango_valido():
    pred = pd.read_csv(OUT / "predicciones_prueba.csv")
    assert pred["probabilidad_buen_rating"].between(0, 1).all()
    assert set(pred["buen_rating"].unique()).issubset({0, 1})
    assert len(pred) > 0


def test_comparacion_modelos_tiene_tres_iteraciones():
    modelos = pd.read_csv(OUT / "comparacion_modelos.csv")
    assert len(modelos) == 3
    for col in ["auc_train", "auc_test", "exactitud", "precision", "recall", "f1"]:
        assert modelos[col].between(0, 1).all()


def test_mejora_iterativa_del_r2():
    """Confirma que el R2 mejora en cada iteracion (justifica la seccion de mejora iterativa)."""
    r2 = pd.read_csv(OUT / "r2_regresion_lineal.csv")
    valores = r2["r2_regresion_lineal"].tolist()
    assert valores == sorted(valores), "El R2 deberia aumentar en cada iteracion del modelo"


def test_resumen_modelo_json_valido():
    with open(OUT / "resumen_modelo.json", encoding="utf-8") as f:
        resumen = json.load(f)
    assert 0 <= resumen["p_valor_hipotesis"] <= 1
    assert resumen["n_productos"] > 0
