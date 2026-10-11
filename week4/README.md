# Semana 4: experimentos con MLflow

En esta actividad pasé el modelo Extra Trees de la semana 3 a un script.
El objetivo es cambiar sus parámetros desde la terminal y comparar los resultados en MLflow.
El notebook se conserva solamente para explorar los datos.

## 1. Preparar el entorno

Desde la carpeta `week4` del repositorio, con Python 3.13.5 (versión utilizada):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activar el entorno con `.venv\Scripts\activate`.
Los CSV están en `data/`, por lo que no hace falta descargarlos.

## 2. Entrenar y repetir los experimentos

```bash
python train.py
python train.py --run-name 02_arboles_100 --n-estimators 100
python train.py --run-name 03_arboles_500 --n-estimators 500
python train.py --run-name 04_variables_0.7 --max-features 0.7
python train.py --run-name 05_hoja_1 --min-samples-leaf 1
python train.py --run-name 06_profundidad_16 --max-depth 16
python train.py --run-name 07_repeticion
```

Son seis configuraciones diferentes y una repetición de la primera.
Se cambia un parámetro a la vez. Cada comando guarda una nueva corrida y puede tardar unos minutos.

| Argumento | Valor inicial | Qué cambia |
| --- | --- | --- |
| `--n-estimators` | 300 | Número de árboles |
| `--max-features` | sqrt | Variables consideradas en cada división |
| `--min-samples-leaf` | 2 | Muestras mínimas en una hoja |
| `--max-depth` | Sin límite | Profundidad máxima de los árboles |
| `--seed` | 42 | Semilla de la partición, los folds y el modelo |
| `--run-name` | 01_referencia | Nombre mostrado en MLflow |

`sqrt` usa aproximadamente la raíz cuadrada del número de variables; `0.7` usa el 70%.
Consultar la ayuda con `python train.py --help`.

## 3. Ver las corridas

Desde esta misma carpeta y con el entorno activo:

```bash
mlflow ui --backend-store-uri sqlite:///.mlflow/week4.db --host 127.0.0.1 --port 5004
```

Abrimos <http://127.0.0.1:5004> y entramos a `TC5061_Semana4_ExtraTrees`.
Seleccionmos las siete corridas y pulsar **Compare**. Comparar los parámetros y las métricas
`cv_roc_auc`, `roc_auc` y `f1`. En el detalle de cada corrida están el modelo,
el archivo de predicciones y la matriz de confusión.

## Cómo funciona

1. Se unen los vinos tintos y blancos y se eliminan duplicados antes de dividir.
2. Se considera calidad alta cuando la calificación es de 7 o más.
3. Se usa 80% para entrenamiento y 20% para prueba, conservando la proporción de clases.
4. Cada configuración se evalúa con cinco grupos de validación dentro del entrenamiento.
5. Se entrena el modelo con todo el conjunto de entrenamiento y se evalúa en prueba.
6. MLflow guarda los parámetros, cinco métricas, dos artefactos y el modelo.

`cv_roc_auc` es el promedio de validación; `roc_auc` se calcula en prueba.
`f1` combina precisión y recall. `recall` indica qué proporción de vinos de calidad alta detectamos.
La mejor configuración se elige por `cv_roc_auc`, no por los resultados de prueba.
La semilla se fija en todos los pasos aleatorios y se usa un hilo para los árboles.
La repetición permite comprobar que las métricas y predicciones son iguales en el mismo entorno.

La base `.mlflow/` y los modelos en `mlartifacts/` se generan localmente y se excluyen de Git.
El CSV y las capturas muestran las corridas de esta entrega no se actualizan automáticamente
al ejecutar nuevos experimentos. Los resultados nuevos se consultan en MLflow.

Datos: Cortez et al. (2009), Wine Quality, UCI Machine Learning Repository.
La descripción y la referencia completa se encuentran en `data/winequality.names`.
