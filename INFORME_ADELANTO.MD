# EcoVision - Informe de adelanto (Semana 2)

**Curso:** Redes Neuronales Artificiales - Unidad 05
**Equipo:** [Nombres]
**Fecha:** [Fecha]
**Repositorio:** [URL del repo]

## 1. Problema

En espacios universitarios se depositan residuos en contenedores incorrectos
por desconocimiento o clasificacion visual imprecisa. Un asistente visual
automatico puede reducir este error mostrando la categoria y el contenedor.

Alcance: 4 categorias visuales (botella plastica, lata, papel/carton,
no identificado) en entorno controlado.

## 2. Arquitectura

Imagen 224x224 -> MobileNetV2 (ImageNet, congelada) -> GlobalAveragePooling2D
-> Dropout(0.3) -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(4, softmax)
Salida: (clase, confianza) -> rules.py (umbral 0.60 + mapeo) -> contenedor.

## 3. Que es red neuronal y que es logica

| Componente | Tipo | Entrada | Salida |
|---|---|---|---|
| MobileNetV2 + Dense(4) | Red neuronal | Imagen 224x224x3 | 4 probabilidades |
| rules.py | Logica | (clase, confianza) | (clase final, contenedor) |

La red solo clasifica. La decision del contenedor es una regla separada.

## 4. Dataset del adelanto

Fuente: TrashNet (2.527 imagenes, 6 clases). Remapeo:
- plastic -> botella_plastica
- metal -> lata
- paper + cardboard -> papel_carton
- glass + trash -> no_identificado

Split: 70% train / 15% val / 15% test.

## 5. Modelo

- Base: MobileNetV2 congelada (ImageNet).
- Cabeza: GAP + Dropout(0.3) + Dense(128, ReLU) + Dropout(0.3) + Dense(4, softmax).
- Optimizador: Adam(1e-4).
- Perdida: categorical cross-entropy.
- Epocas: 15 con early stopping.
- Data augmentation: rotacion 20, zoom 15%, flip horizontal.

## 6. Resultados

## 6. Resultados

Métricas sobre el conjunto de test independiente (377 imágenes, nunca usadas
en entrenamiento ni validación):

- **Accuracy:** 0.86 (86%)
- **Macro-F1:** 0.84

Por clase:

| Clase | Precision | Recall | F1-score | Soporte |
|---|---|---|---|---|
| botella_plastica | 0.85 | 0.79 | 0.82 | 72 |
| lata | 0.77 | 0.90 | 0.83 | 61 |
| no_identificado | 0.79 | 0.74 | 0.76 | 95 |
| papel_carton | 0.95 | 0.96 | 0.96 | 149 |

Matriz de confusión (filas = real, columnas = predicho):

|                | botella | lata | no_id | papel |
|---|---|---|---|---|
| **botella** (72)   | **57** | 2  | 12 | 1   |
| **lata** (61)      | 0      | **55** | 6  | 0   |
| **no_id** (95)     | 9      | 10 | **70** | 6   |
| **papel** (149)    | 1      | 4  | 1  | **143** |

Análisis: papel/cartón es la clase mejor clasificada (F1=0.96).
"No identificado" es la más difícil (F1=0.76) por su heterogeneidad:
al remapear TrashNet agrupamos `glass` y `trash` en esta clase, y muchos
vidrios tienen forma de botella, lo que genera confusión esperable.

## 7. Separacion NN / logica

MobileNetV2 recibe la imagen y devuelve 4 probabilidades. rules.py aplica un
umbral de 0.60 (por debajo -> no_identificado) y mapea a contenedor segun
Resolucion 2184 de 2019.

## 8. Limitaciones actuales

- Dataset provisional (TrashNet), no objetos del campus.
- Sin fine-tuning (base congelada).
- Sin matriz de confusion optimizada.

## 9. Plan para semana 4

1. Recolectar 50-80 fotos propias por clase.
2. Fine-tuning de ultimas 30 capas.
3. Ajustar umbral con curva precision-rechazo.
4. Tabla de 10 pruebas (fondos, luz, objetos desconocidos).
5. Informe final de 4-6 paginas.
6. Video de respaldo de 2-3 min.

## 10. Referencias

- TensorFlow transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning
- MobileNetV2: https://arxiv.org/abs/1801.04381
- TrashNet: https://github.com/garythung/trashnet
- Streamlit: https://docs.streamlit.io/