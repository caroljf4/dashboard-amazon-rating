"""Dashboard interactivo: relacion entre descuento, categoria y rating
de productos en Amazon usa los archivos exportados por
entrenar_modelo.py de la carpeta outputs/ y el dataset limpio.
"""
import base64
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

DATA = pd.read_csv(OUT / "dataset_limpio.csv")
PRED = pd.read_csv(OUT / "predicciones_prueba.csv")
MODELOS = pd.read_csv(OUT / "comparacion_modelos.csv")
R2 = pd.read_csv(OUT / "r2_regresion_lineal.csv")
IMPORTANCIA = pd.read_csv(OUT / "importancia_variables.csv").head(10)
with open(OUT / "resumen_modelo.json", encoding="utf-8") as f:
    RESUMEN = json.load(f)

# Orden para el filtro de rango de descuento
ORDEN_RANGO = ["Bajo", "Medio", "Alto"]

COLORS = {
    "navy": config.COLOR_NAVY,
    "orange": config.COLOR_ORANGE,
    "cyan": config.COLOR_CYAN,
    "gold": config.COLOR_GOLD,
    "red": config.COLOR_RED,
    "paper": "#F5F6F8",
}

# En Binder el dashboard se ve a traves de jupyter-server-proxy (ruta .../proxy/8050/).
# Si existe JUPYTERHUB_SERVICE_PREFIX estamos en Binder y Dash necesita ese prefijo;
# en tu computador la variable no existe y todo funciona igual que antes.
_prefijo_binder = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
if _prefijo_binder:
    app = Dash(
        __name__,
        title=config.TITULO,
        suppress_callback_exceptions=True,
        routes_pathname_prefix="/",
        requests_pathname_prefix=f"{_prefijo_binder}proxy/8050/",
    )
else:
    app = Dash(__name__, title=config.TITULO, suppress_callback_exceptions=True)
server = app.server


def _svg_a_imagen(svg_str: str, clase: str):
    """Convierte un SVG (texto) en una imagen embebida, ya que Dash no soporta
    etiquetas <svg> nativas en su libreria html."""
    b64 = base64.b64encode(svg_str.encode("utf-8")).decode("ascii")
    return html.Img(src=f"data:image/svg+xml;base64,{b64}", className=clase)


def icono_carrito():
    """Icono generico de carrito de compras (trazo libre, no es la marca de Amazon)."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="#FF9900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle>'
        '<path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>'
        "</svg>"
    )
    return _svg_a_imagen(svg, "icono-logo")


def icono_etiqueta():
    """Icono generico de etiqueta de descuento (trazo libre, no es la marca de Amazon)."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="#FF9900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M20.59 13.41 13.41 20.59a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>'
        '<circle cx="7" cy="7" r="1.2" fill="#FF9900" stroke="none"></circle>'
        "</svg>"
    )
    return _svg_a_imagen(svg, "icono-etiqueta")


def tarjeta(titulo: str, value_id: str, nota: str, emoji: str = ""):
    return html.Div(
        [
            html.P(
                [html.Span(emoji, className="kpi-title-emoji") if emoji else "", titulo],
                className="kpi-title",
            ),
            html.H2(id=value_id),
            html.P(nota, className="kpi-note"),
        ],
        className="kpi-card",
    )


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.P(config.EYEBROW, className="eyebrow"),
                        html.H1([icono_carrito(), config.TITULO]),
                        html.P(config.SUBTITULO, className="subtitle"),
                    ]
                ),
                html.Div(config.BADGE, className="badge"),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Aside(
                    [
                        html.H3([icono_etiqueta(), " ", config.TITULO_FILTROS]),
                        html.Label(config.LABEL_CATEGORIA),
                        dcc.Dropdown(
                            sorted(DATA["categoria_principal"].unique()),
                            multi=True,
                            id="categoria",
                            placeholder="Todas las categorias",
                        ),
                        html.Label(config.LABEL_RANGO_DESCUENTO),
                        dcc.Dropdown(
                            ORDEN_RANGO, multi=True, id="rango-descuento", placeholder="Todos los rangos"
                        ),
                        html.Label(config.LABEL_RANGO_DESCUENTO_SLIDER),
                        dcc.RangeSlider(
                            0, 100, 5, value=[0, 100],
                            marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                            id="rango-descuento-slider",
                        ),
                        html.Hr(),
                        html.Label(config.LABEL_UMBRAL),
                        dcc.Slider(
                            0.30, 0.90, 0.05, value=0.50,
                            marks={x: f"{x:.1f}" for x in [0.3, 0.5, 0.7, 0.9]},
                            id="umbral",
                        ),
                        html.P(config.TEXTO_AYUDA_UMBRAL, className="help"),
                        html.Button(config.TEXTO_BOTON_RESET, id="reset", n_clicks=0),
                    ],
                    className="sidebar",
                ),
                html.Main(
                    [
                        html.Div(
                            [
                                tarjeta(config.KPI_1_TITULO, "kpi-n", config.KPI_1_NOTA, emoji=config.EMOJI_KPI_1),
                                tarjeta(config.KPI_2_TITULO, "kpi-rating", config.KPI_2_NOTA, emoji=config.EMOJI_KPI_2),
                                tarjeta(config.KPI_3_TITULO, "kpi-buen", config.KPI_3_NOTA, emoji=config.EMOJI_KPI_3),
                                tarjeta(config.KPI_4_TITULO, "kpi-descuento", config.KPI_4_NOTA, emoji=config.EMOJI_KPI_4),
                            ],
                            className="kpi-grid",
                        ),
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label=config.TAB_1,
                                    children=[
                                        html.Div(
                                            [dcc.Graph(id="cat-chart"), dcc.Graph(id="rango-chart")],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="scatter-chart"), dcc.Graph(id="prob-chart")],
                                            className="grid-2",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label=config.TAB_2,
                                    children=[
                                        html.Div(id="metric-strip", className="metric-strip"),
                                        html.Div(
                                            [dcc.Graph(id="confusion-chart"), dcc.Graph(id="modelos-chart")],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="r2-chart"), dcc.Graph(id="importancia-chart")],
                                            className="grid-2",
                                        ),
                                        html.Div(id="narrativa", className="narrative"),
                                    ],
                                ),
                                dcc.Tab(
                                    label=config.TAB_3,
                                    children=[
                                        html.P(config.TEXTO_TABLA_CASOS, className="tab-intro"),
                                        dash_table.DataTable(
                                            id="tabla-casos",
                                            page_size=12,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "fontFamily": "Arial",
                                                "fontSize": 12,
                                                "padding": "8px",
                                                "textAlign": "left",
                                            },
                                            style_header={
                                                "backgroundColor": COLORS["navy"],
                                                "color": "white",
                                                "fontWeight": "bold",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"filter_query": "{nivel_riesgo} = Alto"},
                                                    "backgroundColor": "#FCE8E8",
                                                }
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        html.Footer(config.TEXTO_FOOTER),
                    ],
                    className="content",
                ),
            ],
            className="shell",
        ),
    ],
    className="app",
)


@app.callback(
    Output("categoria", "value"),
    Output("rango-descuento", "value"),
    Output("umbral", "value"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True,
)
def restablecer(_):
    return [], [], 0.50


@app.callback(
    Output("kpi-n", "children"),
    Output("kpi-rating", "children"),
    Output("kpi-buen", "children"),
    Output("kpi-descuento", "children"),
    Output("cat-chart", "figure"),
    Output("rango-chart", "figure"),
    Output("scatter-chart", "figure"),
    Output("prob-chart", "figure"),
    Output("metric-strip", "children"),
    Output("confusion-chart", "figure"),
    Output("modelos-chart", "figure"),
    Output("r2-chart", "figure"),
    Output("importancia-chart", "figure"),
    Output("narrativa", "children"),
    Output("tabla-casos", "data"),
    Output("tabla-casos", "columns"),
    Input("categoria", "value"),
    Input("rango-descuento", "value"),
    Input("rango-descuento-slider", "value"),
    Input("umbral", "value"),
)
def actualizar(categorias, rangos, rango_slider, umbral):
    template = "plotly_white"

    # *** Filtrado del dataset completo para KPIs y panorama ***
    d = DATA.copy()
    if categorias:
        d = d[d["categoria_principal"].isin(categorias)]
    if rangos:
        d = d[d["rango_descuento"].isin(rangos)]
    if rango_slider:
        d = d[d["discount_percentage"].between(rango_slider[0], rango_slider[1])]

    # *** Filtrado del conjunto de prueba para rendimiento y tabla de casos ***
    p = PRED.copy()
    if categorias:
        p = p[p["categoria_principal"].isin(categorias)]
    if rangos:
        p = p[p["rango_descuento"].isin(rangos)]
    if rango_slider:
        p = p[p["discount_percentage"].between(rango_slider[0], rango_slider[1])]

    if d.empty:
        vacia = go.Figure().update_layout(template=template, title="No hay datos con estos filtros")
        return (
            "0", "--", "--", "--", vacia, vacia, vacia, vacia, [], vacia, vacia, vacia, vacia,
            [html.P("No hay datos para mostrar con esta combinacion de filtros.")], [], [],
        )

    # *** KPIs ***
    kpi_n = f"{len(d):,}"
    kpi_rating = f"{d['rating'].mean():.2f}"
    kpi_buen = f"{d['buen_rating'].mean():.1%}"
    kpi_descuento = f"{d['discount_percentage'].mean():.1f}%"

    # *** Panorama: rating por categoria ***
    por_cat = (
        d.groupby("categoria_principal", as_index=False)
        .agg(rating_promedio=("rating", "mean"), productos=("id", "count"))
        .sort_values("rating_promedio")
    )
    fig_cat = px.bar(
        por_cat, x="rating_promedio", y="categoria_principal", orientation="h",
        text=por_cat["rating_promedio"].map(lambda x: f"{x:.2f}"),
        title=config.TITULO_CAT_CHART,
        color="rating_promedio", color_continuous_scale=["#FCE9C9", COLORS["orange"]],
    )
    fig_cat.update_layout(template=template, coloraxis_showscale=False, xaxis_title="Rating", yaxis_title="")

    # *** Panorama: rating por rango de descuento ***
    por_rango = (
        d.groupby("rango_descuento", as_index=False, observed=True)
        .agg(rating_promedio=("rating", "mean"))
    )
    por_rango["rango_descuento"] = pd.Categorical(por_rango["rango_descuento"], categories=ORDEN_RANGO, ordered=True)
    por_rango = por_rango.sort_values("rango_descuento")
    fig_rango = px.bar(
        por_rango, x="rango_descuento", y="rating_promedio",
        title=config.TITULO_RANGO_CHART,
        color_discrete_sequence=[COLORS["cyan"]],
        text=por_rango["rating_promedio"].map(lambda x: f"{x:.2f}"),
    )
    fig_rango.update_layout(template=template, xaxis_title="Rango de descuento", yaxis_title="Rating promedio")

    # -*** Panorama: scatter descuento vs rating ***
    muestra = d.sample(min(800, len(d)), random_state=7) if len(d) else d
    top5_cat_muestra = d["categoria_principal"].value_counts().head(5).index
    muestra_color = muestra["categoria_principal"].where(
        muestra["categoria_principal"].isin(top5_cat_muestra), "Otras categorias"
    )
    fig_scatter = px.scatter(
        muestra, x="discount_percentage", y="rating",
        color=muestra_color, opacity=0.55, title=config.TITULO_SCATTER_CHART,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_scatter.update_layout(template=template, xaxis_title="Descuento (%)", yaxis_title="Rating", legend_title="Categoria")

    # *** Panorama: distribucion de probabilidades: conjunto de prueba ***
    if len(p):
        fig_prob = px.histogram(
            p, x="probabilidad_buen_rating",
            color=p["buen_rating"].map({0: "Mal rating real", 1: "Buen rating real"}),
            nbins=20, barmode="overlay", opacity=0.65,
            title=config.TITULO_PROB_CHART,
            color_discrete_map={"Mal rating real": COLORS["red"], "Buen rating real": COLORS["cyan"]},
        )
        fig_prob.add_vline(x=umbral, line_dash="dash", line_color=COLORS["gold"])
        fig_prob.update_layout(template=template, xaxis_title="Probabilidad de buen rating", legend_title="")
    else:
        fig_prob = go.Figure().update_layout(template=template, title="Sin datos de prueba en esta seleccion")

    # *** Rendimiento del modelo: segun umbral, sobre el conjunto de prueba filtrado ***
    if len(p) and p["buen_rating"].nunique() == 2:
        y_true = p["buen_rating"].to_numpy()
        y_pred = (p["probabilidad_buen_rating"].to_numpy() >= umbral).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        strip = [
            html.Div([html.Span("Exactitud"), html.Strong(f"{acc:.1%}")]),
            html.Div([html.Span("Precision"), html.Strong(f"{prec:.1%}")]),
            html.Div([html.Span("Sensibilidad: recall"), html.Strong(f"{rec:.1%}")]),
            html.Div([html.Span("F1"), html.Strong(f"{f1:.1%}")]),
        ]
        especificidad = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else float("nan")
    else:
        cm = np.zeros((2, 2), dtype=int)
        strip = [html.Div([html.Span("Sin ambas clases en esta seleccion"), html.Strong("--")])]
        rec = prec = especificidad = float("nan")

    fig_cm = go.Figure(
        go.Heatmap(
            z=cm, x=["Predice mal rating", "Predice buen rating"],
            y=["Real mal rating", "Real buen rating"],
            text=cm, texttemplate="%{text}",
            colorscale=[[0, "#EAF1F7"], [1, COLORS["cyan"]]], showscale=False,
        )
    )
    fig_cm.update_layout(template=template, title=f"{config.TITULO_CONFUSION_CHART} (umbral {umbral:.0%})")

    # *** Comparacion de las 3 iteraciones del modelo ***
    modelos_largo = MODELOS.melt(
        id_vars="modelo", value_vars=["auc_test", "exactitud", "f1"], var_name="metrica", value_name="valor"
    )
    modelos_largo["metrica"] = modelos_largo["metrica"].map(
        {"auc_test": "ROC-AUC (prueba)", "exactitud": "Exactitud", "f1": "F1"}
    )
    fig_modelos = px.bar(
        modelos_largo, x="modelo", y="valor", color="metrica", barmode="group",
        title=config.TITULO_MODELOS_CHART,
        color_discrete_sequence=[COLORS["navy"], COLORS["cyan"], COLORS["orange"]],
    )
    fig_modelos.update_layout(template=template, yaxis_tickformat=".0%", xaxis_title="")

    # *** R2 de la regresion lineal en las 3 iteraciones ***
    fig_r2 = px.bar(
        R2, x="modelo", y="r2_regresion_lineal", title=config.TITULO_R2_CHART,
        color_discrete_sequence=[COLORS["orange"]], text=R2["r2_regresion_lineal"].map(lambda x: f"{x:.3f}"),
    )
    fig_r2.update_layout(template=template, yaxis_title="R²", xaxis_title="")

    # *** Importancia de variables ***
    imp = IMPORTANCIA.sort_values("importancia")
    fig_imp = px.bar(
        imp, x="importancia", y="variable", orientation="h", color="signo",
        title=config.TITULO_IMPORTANCIA_CHART,
        color_discrete_map={"Aumenta": COLORS["cyan"], "Reduce": COLORS["red"]},
    )
    fig_imp.update_layout(template=template, xaxis_title="Magnitud del coeficiente (estandarizado)", yaxis_title="")

    # -*** Narrativa ejecutiva ***
    narrativa = [
        html.H3(config.NARRATIVA_TITULO),
        html.P(
            f"La selección analizada está compuesta por {len(d):,} productos en promedio estos productos tienen una calificación de "
            f"{d['rating'].mean():.2f} y un descuento del {d['discount_percentage'].mean():.1f}%."
        ),
        html.P(
            f"Al utilizar un umbral del {umbral:.0%}, el modelo que también considera la categoría del producto logró identificar correctamente todos los casos positivos con una sensibilidad del "
            f"{rec:.1%} y una especificidad de {especificidad:.1%} sobre el conjunto de prueba filtrado."
            if not np.isnan(rec) else "La seleccion actual no permite calcular sensibilidad y especificidad."
        ),
        html.P(
            f"La prueba de hipótesis mostró que existe una diferencia estadísticamente significativa entre la calificación promedio de los productos con "
            f"descuentos altos y bajos: p = {RESUMEN['p_valor_hipotesis']:.2e} sin embargo el tamaño de esta diferencia fue pequeño a moderado como indica el valor de "
            f"Cohen's d = {RESUMEN['cohen_d']:.3f}."
        ),
        html.P(
            "La incorporación de la categoría del producto fue el cambio que más mejoró el modelo ya que al incluir esta variable el R² de la regresión lineal aumentó de 0,022 a 0,111 mientras que el ROC-AUC del modelo de clasificación pasó de 0,57 a 0,69 indicando que el conocer la categoría del producto ayuda a explicar y predecir mejor los resultados. "
            
        ),
        html.P(config.NARRATIVA_NOTA_FINAL, className="help"),
    ]

    # *** Tabla de casos ***
    cols = [
        "id", "product_name", "categoria_principal", "rango_descuento",
        "discount_percentage", "rating", "probabilidad_buen_rating", "nivel_riesgo",
    ]
    tabla = p.sort_values("probabilidad_buen_rating", ascending=True)[cols].head(150).copy()
    tabla["probabilidad_buen_rating"] = tabla["probabilidad_buen_rating"].round(3)
    columnas = [{"name": c.replace("_", " ").title(), "id": c} for c in cols]

    return (
        kpi_n, kpi_rating, kpi_buen, kpi_descuento,
        fig_cat, fig_rango, fig_scatter, fig_prob,
        strip, fig_cm, fig_modelos, fig_r2, fig_imp, narrativa,
        tabla.to_dict("records"), columnas,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
