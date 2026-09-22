# Guia de entrega y correspondencia con la actividad de la etapa de transferencia

## Entregable 1. Informe del proceso

- `1. Actividad Transferencia PPBD 2 Carol Lopez` (fuera de esta carpeta, en la raiz de la entrega): problema,
  decisiones, datos, procesamiento, iteraciones del modelo, hiperparametros,
  regularizacion, metricas, dashboard, conclusiones y limitaciones.
- `anexos/`: notebooks y PDFs de las etapas de contextualizacion (EDA) y
  profundizacion (contraste de hipotesis y regresiones).

## Entregable 2. Carpeta del dashboard (esta carpeta)

- `app.py`: dashboard Dash con graficos, tabla, filtros y callbacks.
- `assets/style.css`: apariencia y diseno adaptable, tema Amazon.
- `data/amazon.csv`: dataset original.
- `outputs/`: dataset limpio, predicciones de prueba, comparacion de las 3
  iteraciones del modelo, R² por iteracion e importancia de variables.
- `entrenar_modelo.py`: reproduce la limpieza, el entrenamiento y la
  exportacion de resultados.
- `requirements.txt`, `environment.yml`, `Procfile`, `binder/start`: configuracion
  de entorno y despliegue.
- `tests/`: pruebas automatizadas de integridad de datos y resultados.

## Entregable 3. Archivo con enlaces

- `ENLACES.txt` (en la raiz de la entrega): plantilla para el repositorio de
  GitHub y el enlace de Binder.

## Inicio rapido

1. Cree y active un entorno virtual.
2. Ejecute `pip install -r requirements.txt`.
3. Ejecute `python entrenar_modelo.py` (genera la carpeta `outputs/`).
4. Ejecute `python app.py`.
5. Abra `http://127.0.0.1:8050`.
