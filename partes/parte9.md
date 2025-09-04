# Parte 9: Generación

Una vez que el modelo ha sido entrenado y hemos guardado los parámetros que mejor generalizan, llega el momento de ponerlo a prueba en su tarea principal: la generación de texto, que es un proceso fundamentalmente diferente al entrenamiento, ya que aquí el modelo no compara su salida con una respuesta correcta, sino que debe construir una secuencia nueva, carácter a carácter o token a token, usando únicamente lo que ha aprendido y el contexto que le proporcionamos.

<p align="center">
  <img src="../assets/output-text.png" width="400">
</p>
<p align="center"><i>Ejemplo de generación de texto, donde el modelo va completando la secuencia paso a paso.</i></p>

## Diferencias clave con el entrenamiento

Durante el entrenamiento, el modelo ve secuencias completas y aprende a predecir el siguiente token en cada posición, pero en la generación, el proceso es autoregresivo, es decir, el modelo genera un token, lo añade al contexto y vuelve a predecir el siguiente, repitiendo este ciclo hasta alcanzar el límite definido o hasta que se genere un token de parada, así que la generación es una concatenación progresiva de caracteres o tokens, donde cada predicción depende de todas las anteriores.


## Temperatura: controlando la aleatoriedad

Uno de los parámetros más importantes en la generación es la **temperatura**, que ajusta la distribución de probabilidades antes de muestrear el siguiente token, una temperatura baja (por ejemplo, 0.7) hace que el modelo sea más conservador y elija tokens con mayor probabilidad, generando textos más predecibles, mientras que una temperatura alta (por ejemplo, 1.2) hace que la distribución sea más plana y el modelo se arriesgue más, generando textos más variados pero también más caóticos.



## Top-k y Top-p: filtrando las opciones

Además de la temperatura, existen técnicas como **top-k** y **top-p (nucleus sampling)** para controlar la diversidad de la generación, en top-k, solo se consideran los k tokens con mayor probabilidad y se descartan el resto, lo que limita las opciones y evita que el modelo elija tokens poco probables, en top-p, se seleccionan los tokens más probables cuya suma de probabilidades supera un umbral p (por ejemplo, 0.9), adaptando dinámicamente el número de opciones según la distribución, ambas técnicas ayudan a equilibrar creatividad y coherencia en el texto generado.



## Límite de generación: block size y truncamiento

El modelo solo puede generar secuencias hasta un cierto límite, definido por el **block size**, que es la longitud máxima de contexto que puede manejar, si intentamos generar más allá de este límite, el modelo solo tendrá en cuenta los últimos tokens dentro de ese tamaño, lo que significa que la memoria del modelo es limitada y no puede recordar todo el historial, así que la generación se detiene cuando se alcanza el block size, cuando se genera un token especial de parada o cuando se cumple el número máximo de tokens definido por el usuario.


De esta forma, la generación de texto en un modelo GPT es un proceso controlado, donde cada decisión depende de los parámetros de muestreo y de los límites arquitectónicos, y donde el equilibrio entre creatividad y coherencia se ajusta cuidadosamente para obtener los mejores resultados posibles según la aplicación.