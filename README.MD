# EcoVision - Clasificador visual de residuos

Proyecto del curso Redes Neuronales Artificiales (Unidad 05).
Clasifica imagenes de residuos en 4 categorias y sugiere el contenedor.

## Clases
- botella_plastica
- lata
- papel_carton
- no_identificado

## Arquitectura
Imagen 224x224 -> MobileNetV2 (congelada) -> GAP -> Dense(128) -> softmax(4)
Salida: (clase, confianza) -> rules.py -> contenedor sugerido

## Instalacion
pip install -r requirements.txt

## Uso
streamlit run app.py

## Plan B
python imshow.py dataset/test

## Entrenamiento
python remap.py
python train_model.py

## Autores
[Joás Alvarez, Nestor Granadillo, Luis Miranda, Breiner Rosellón, Jesús cardenas, Jesús Silva, Ramsés Sepúlveda]