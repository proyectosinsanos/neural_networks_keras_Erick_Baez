# README - Entrenamiento de Red Neuronal con MNIST

## Descripción
Este proyecto implementa una red neuronal para el reconocimiento de dígitos escritos a mano utilizando la base de datos MNIST. La red es construida con Keras y entrenada para clasificar imágenes de dígitos del 0 al 9. El modelo utiliza capas densas con activaciones ReLU y Softmax.

## Requisitos
Antes de ejecutar el código, asegúrese de tener instaladas las siguientes bibliotecas:

- `numpy`
- `keras`
- `matplotlib`

Puede instalarlas utilizando:
```bash
pip install numpy keras matplotlib
```

## Estructura del Código
El código sigue los siguientes pasos:
1. **Carga de Datos:** Se obtiene el conjunto de entrenamiento y prueba de la base de datos MNIST.
2. **Visualización de Ejemplo:** Se muestra una imagen de entrenamiento como referencia.
3. **Preprocesamiento:** Se normalizan las imágenes y se convierten las etiquetas a formato one-hot encoding.
4. **Construcción del Modelo:** Se define una red neuronal con una capa de entrada, una capa oculta con 512 neuronas y una capa de salida con 10 neuronas.
5. **Compilación del Modelo:** Se utiliza `rmsprop` como optimizador y `categorical_crossentropy` como función de pérdida.
6. **Entrenamiento:** Se entrena el modelo con 8 épocas y un tamaño de batch de 128.
7. **Evaluación:** Se mide la precisión del modelo en el conjunto de prueba.

## Ejecución
Para entrenar y evaluar el modelo, ejecute:
```bash
python main.py
```


## Resultados
Al finalizar, se mostrará la precisión alcanzada en el conjunto de prueba. También se visualizará un ejemplo de imagen de entrenamiento.