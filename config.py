# *** Encabezado (parte superior, franja oscura) *** 
EYEBROW = "PROYECTO PARA CIENCIA DE DATOS E-COMMERCE"
TITULO = "¿El descuento predice el rating de un producto de Amazon?"
SUBTITULO = (
    "Exploracion, contraste de hipotesis y modelos de clasificacion sobre el "
    "Amazon Sales Dataset con 1,462 productos."
)
BADGE = "DASH + PLOTLY"

# *** Sidebar: filtros ***
TITULO_FILTROS = "Filtros interactivos"
LABEL_CATEGORIA = "Categoria del producto"
LABEL_RANGO_DESCUENTO = "Rango de descuento"
LABEL_RANGO_DESCUENTO_SLIDER = "Rango de descuento (%)"
LABEL_UMBRAL = "Umbral de clasificacion con buen rating"
TEXTO_AYUDA_UMBRAL = (
    "El umbral cambia la regla que convierte la probabilidad en "
    "'buen rating' o 'mal rating', este no reentrena el modelo."
)
TEXTO_BOTON_RESET = "Restablecer filtros"

# *** Tarjetas KPI ***
KPI_1_TITULO, KPI_1_NOTA, EMOJI_KPI_1 = "Productos", "en la seleccion actual", "🛒"
KPI_2_TITULO, KPI_2_NOTA, EMOJI_KPI_2 = "Rating promedio", "escala 1 a 5", "⭐"
KPI_3_TITULO, KPI_3_NOTA, EMOJI_KPI_3 = "Buen rating", "rating >= 4.0", "✅"
KPI_4_TITULO, KPI_4_NOTA, EMOJI_KPI_4 = "Descuento promedio", "porcentaje", "🏷️"

# *** Nombres de pestañas ***
TAB_1 = "Panorama"
TAB_2 = "Rendimiento del modelo"
TAB_3 = "Casos: conjunto de prueba"

# *** Titulos de los graficos ***
TITULO_CAT_CHART = "Rating promedio por categoria"
TITULO_RANGO_CHART = "Rating promedio por rango de descuento"
TITULO_SCATTER_CHART = "Descuento vs. rating clasificado por categoria"
TITULO_PROB_CHART = "Distribucion de probabilidades, conjunto de prueba"
TITULO_CONFUSION_CHART = "Matriz de confusion"
TITULO_MODELOS_CHART = "Comparacion de las 3 iteraciones del modelo"
TITULO_R2_CHART = "R\u00b2 de la regresion lineal por iteracion"
TITULO_IMPORTANCIA_CHART = "Importancia de variables: modelo con categoria"

# *** Texto de la pestaña "Casos" ***
TEXTO_TABLA_CASOS = (
    "Productos del conjunto de prueba ordenados por probabilidad de mal rating "
    "de mayor riesgo primero, el modelo usado es el de la iteracion 3 con categoria del producto."
)

# *** Narrativa ejecutiva: pestaña "Rendimiento del modelo" ***
NARRATIVA_TITULO = "Lectura ejecutiva"
NARRATIVA_NOTA_FINAL = (
    "Estos resultados son asociaciones, no relaciones de causalidad, al mover el umbral cambia "
    "el equilibrio entre detectar mas productos de mal rating y generar mas falsas alertas."
)

# *** Pie de pagina ***
TEXTO_FOOTER = (
    "Fuente: Amazon Sales Dataset descargado de Kaggle. Analisis desarrollado en las etapas de "
    "contextualizacion y profundizacion del curso Programacion para Ciencia de Datos II."
)

# *** Colores asociados al tema y la identidad de marca de Amazon ***
COLOR_NAVY = "#131921"      
COLOR_ORANGE = "#FF9900"    
COLOR_CYAN = "#146EB4"      
COLOR_GOLD = "#F4B942"      
COLOR_RED = "#D95D5D"
