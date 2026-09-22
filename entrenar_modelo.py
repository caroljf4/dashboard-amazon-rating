"""Hola por favor procesa el dataset de Amazon, entrena las 3 iteraciones del modelo
(descuento simple, regresion multiple, y con categoria) y exporta los
archivos que usa el dashboard: predicciones, comparacion de modelos
e importancia de variables.

Ejecuta una sola vez o cuando cambien los datos/modelo:
    python entrenar_modelo.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def limpiar_precio(columna: pd.Series) -> pd.Series:
    return (
        columna.astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def cargar_y_limpiar() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "amazon.csv")

    df["discounted_price"] = limpiar_precio(df["discounted_price"])
    df["actual_price"] = limpiar_precio(df["actual_price"])
    df["discount_percentage"] = (
        df["discount_percentage"].astype(str).str.replace("%", "", regex=False).astype(float)
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(
        df["rating_count"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df["categoria_principal"] = df["category"].str.split("|").str[0]

    df = df.dropna(
        subset=["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
    )
    df = df.drop_duplicates()

    df["rango_descuento"] = pd.cut(
        df["discount_percentage"], bins=[-1, 30, 60, 100], labels=["Bajo", "Medio", "Alto"]
    )
    df["buen_rating"] = (df["rating"] >= 4.0).astype(int)

    # id unico y legible para el dashboard
    df = df.reset_index(drop=True)
    df["id"] = "P" + df.index.astype(str).str.zfill(4)

    return df


def entrenar_logistica(X: pd.DataFrame, y: pd.Series, nombre: str, seed: int = 42):
    """Entrena una regresion logistica y devuelve metricas de train y test (para ver sobreajuste).

    Las variables numericas se estandarizan (media 0, desviacion 1) antes de entrenar:
    esto evita problemas de convergencia cuando las variables tienen escalas muy distintas
    (ej. precios en miles de rupias vs. porcentajes de 0 a 100) y no cambia la interpretacion
    de cual variable pesa mas, solo hace que el optimizador numerico converja de forma estable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    escalador = StandardScaler()
    X_train_esc = pd.DataFrame(escalador.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_esc = pd.DataFrame(escalador.transform(X_test), columns=X_test.columns, index=X_test.index)

    modelo = LogisticRegression(max_iter=2000)
    modelo.fit(X_train_esc, y_train)
    X_train, X_test = X_train_esc, X_test_esc

    prob_train = modelo.predict_proba(X_train)[:, 1]
    prob_test = modelo.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)
    pred_train = (prob_train >= 0.5).astype(int)

    metricas = {
        "modelo": nombre,
        "auc_train": roc_auc_score(y_train, prob_train),
        "auc_test": roc_auc_score(y_test, prob_test),
        "exactitud": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
    }
    return modelo, X_test, y_test, prob_test, metricas


def main():
    df = cargar_y_limpiar()

    # *** ITERACION 1: solo descuento ***
    X1 = df[["discount_percentage"]]
    y = df["buen_rating"]
    modelo1, X1_test, y1_test, prob1_test, met1 = entrenar_logistica(X1, y, "1. Solo descuento")

    # *** ITERACION 2: descuento + precio + resenas ***
    features_num = ["discounted_price", "actual_price", "discount_percentage", "rating_count"]
    X2 = df[features_num]
    modelo2, X2_test, y2_test, prob2_test, met2 = entrenar_logistica(X2, y, "2. + Variables numericas")

    # *** ITERACION 3: + categoria (dummies) ***
    dummies = pd.get_dummies(df["categoria_principal"], prefix="cat", drop_first=True)
    X3 = pd.concat([df[features_num], dummies], axis=1)
    modelo3, X3_test, y3_test, prob3_test, met3 = entrenar_logistica(X3, y, "3. + Categoria del producto")

    comparacion = pd.DataFrame([met1, met2, met3])
    comparacion.to_csv(OUT_DIR / "comparacion_modelos.csv", index=False)

    # *** Regresion lineal (para el R^2, referencia del dashboard) ***
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        df[features_num], df["rating"], test_size=0.30, random_state=42
    )
    reg_simple = LinearRegression().fit(df[["discount_percentage"]].loc[Xl_train.index], yl_train)
    r2_simple = r2_score(
        yl_test, reg_simple.predict(df[["discount_percentage"]].loc[Xl_test.index])
    )
    reg_mult = LinearRegression().fit(Xl_train, yl_train)
    r2_mult = r2_score(yl_test, reg_mult.predict(Xl_test))
    Xl3 = pd.concat([df[features_num], dummies], axis=1)
    Xl3_train, Xl3_test = Xl3.loc[Xl_train.index], Xl3.loc[Xl_test.index]
    reg_cat = LinearRegression().fit(Xl3_train, yl_train)
    r2_cat = r2_score(yl_test, reg_cat.predict(Xl3_test))

    r2_resumen = pd.DataFrame(
        {
            "modelo": ["1. Solo descuento", "2. + Variables numericas", "3. + Categoria del producto"],
            "r2_regresion_lineal": [r2_simple, r2_mult, r2_cat],
        }
    )
    r2_resumen.to_csv(OUT_DIR / "r2_regresion_lineal.csv", index=False)

    # *** Predicciones del MEJOR modelo (iteracion 3) para el dashboard ***
    pred_out = df.loc[X3_test.index, [
        "id", "product_name", "categoria_principal", "rango_descuento",
        "discounted_price", "actual_price", "discount_percentage",
        "rating", "rating_count", "buen_rating",
    ]].copy()
    pred_out["probabilidad_buen_rating"] = prob3_test
    pred_out["nivel_riesgo"] = pd.cut(
        1 - pred_out["probabilidad_buen_rating"],
        bins=[-0.001, 0.15, 0.35, 1.0],
        labels=["Bajo", "Medio", "Alto"],
    )
    pred_out.to_csv(OUT_DIR / "predicciones_prueba.csv", index=False)

    # *** Importancia de variables (coeficientes del modelo 3) ***
    coefs = pd.Series(modelo3.coef_[0], index=X3.columns)
    importancia = (
        coefs.abs()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "variable", 0: "importancia"})
    )
    importancia.columns = ["variable", "importancia"]
    importancia["signo"] = coefs.loc[importancia["variable"]].apply(lambda v: "Aumenta" if v > 0 else "Reduce").values
    importancia.to_csv(OUT_DIR / "importancia_variables.csv", index=False)

    # *** Contraste de hipotesis (para mostrar en el dashboard tambien) ***
    grupo_alto = df.loc[df["rango_descuento"] == "Alto", "rating"]
    grupo_bajo = df.loc[df["rango_descuento"] == "Bajo", "rating"]
    t_stat, p_val = stats.ttest_ind(grupo_alto, grupo_bajo, equal_var=False)
    diff = grupo_alto.mean() - grupo_bajo.mean()
    n1, n2 = len(grupo_alto), len(grupo_bajo)
    s1, s2 = grupo_alto.std(ddof=1), grupo_bajo.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohen_d = diff / sp

    resumen_json = {
        "n_productos": int(len(df)),
        "rating_promedio": float(df["rating"].mean()),
        "descuento_promedio": float(df["discount_percentage"].mean()),
        "pct_buen_rating": float(df["buen_rating"].mean()),
        "t_stat_hipotesis": float(t_stat),
        "p_valor_hipotesis": float(p_val),
        "cohen_d": float(cohen_d),
        "r2_mejor_modelo": float(r2_cat),
        "auc_mejor_modelo": float(met3["auc_test"]),
    }
    import json

    with open(OUT_DIR / "resumen_modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, indent=2, ensure_ascii=False)

    # Se guarda tambien el dataset limpio completo, para los graficos exploratorios del dashboard
    df.to_csv(OUT_DIR / "dataset_limpio.csv", index=False)

    print("Listo. Archivos exportados en:", OUT_DIR)
    print(comparacion)
    print(r2_resumen)
    print(resumen_json)


if __name__ == "__main__":
    main()
