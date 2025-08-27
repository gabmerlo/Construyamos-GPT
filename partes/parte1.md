# Parte 1: Primeros Pasos (Google Collab)

**Si ya estáis familiarizados con Google Collab, no hace falta que hagáis esta sección.**


Escribo esta sección para aquellos que no estáis familiarizados con Google Collab. que es la herramienta principal que vamos a utilizar para aprender a construir un GPT, que no os preocupéis, el uso que le tendréis que dar no es nada complicado y tendréis hecho ya todo.

Necesitamos Google Collab tanto para ejecutar el código que usaremos, como para procesar los datos que tenemos. No será necesario programar ni saber trabajar con datos, pero sí que es necesario que sepáis cómo ejecutar código dentro de Collab y subir los archivos que os proporcionaré.

Para aprender estas dos cosas que necesitaremos, lo haremos juntos dentro de la guía que he preparado para vosotros en el mismo Collab, podéis acceder a través del siguiente enlace:

[https://colab.research.google.com/drive/15COK7eW-1l0yaVwqeBM78nJ56mHGJmlf?usp=sharing](https://colab.research.google.com/drive/15COK7eW-1l0yaVwqeBM78nJ56mHGJmlf?usp=sharing)


Dentro de ese Collab, os guiaré paso a paso para que aprendáis dos cosas fundamentales:

**1. Ejecutar una celda de código.**  
**2. Subir un archivo con nuestros datos.**

Aunque todo está explicado dentro del propio Collab, os dejo aquí un resumen de lo que encontraréis para que os sirva de guía.

## Tarea 1: Ejecutar vuestra primera celda de código

Sé que la interfaz puede parecer difícil la primera vez que la ves, pero es más simple de lo que parece. Veréis que el contenido está organizado en 'celdas', que son esos rectángulos que contienen texto o código.

Nuestro primer objetivo es ejecutar una de estas celdas de código, y para ello, solo tenéis que poner el cursor sobre la esquina superior izquierda de la celda de código, donde os aparecerá un botón con forma de Play.

<br>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/gabmerlo/assets-gpt@main/assets/ejecutar.png" width="350">
</p>


<br>

También podéis hacerlo haciendo clic en la celda y pulsando **Ctrl + Enter**.

Cuando lo hagáis, veréis que ocurren tres cosas:
1.  **La primera vez tarda un poco**: Collab está iniciándose.
2.  **Aparece un tick verde**: A la izquierda de la celda, esto indica que se ha ejecutado correctamente.
3.  **Se imprime un mensaje**: En nuestro caso, aparecerá el mensaje "Buenos días Sancho" justo debajo de la celda.

Con esto ya habréis dominado la primera habilidad esencial para construir nuestro GPT.

## Tarea 2: Subir nuestros datos

Para poder entrenar a ***sancho-mini***, necesitamos darle el texto de Cervantes, por lo que tendremos que saber cómo subir este archivo a Google Collab. (Que no os preocupéis, no es tan intimidante como suena, es solo subir un archivo de texto).

Primero, necesitáis descargar el archivo que usaremos. Podéis encontrarlo aquí:

[**Descargar datos_sancho_mini.txt**](https://github.com/gabmerlo/Construyamos-GPT/blob/main/data/datos_sancho_mini.txt)

Para descargarlo, solo tenéis que hacer clic en el botón de descarga que aparece en la barra superior del archivo, como se ve en esta imagen:

<br>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/gabmerlo/assets-gpt@main/assets/descarga-archivo.png" width="440">
</p>



<br>

Una vez lo tengáis en vuestro ordenador, el siguiente paso es subirlo a Collab. Para ello, tenéis que:

<br>

1.  Hacer clic en el icono de la carpeta en la barra de la izquierda para abrir el gestor de archivos.

<br>


<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/gabmerlo/assets-gpt@main/assets/archivos.png" width="230">
</p>


<br>


2.  Una vez abierto, podéis simplemente **arrastrar el archivo** `datos_sancho_mini.txt` desde vuestro ordenador a ese panel, o usar el botón de subir archivo.


<br>



<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/gabmerlo/assets-gpt@main/assets/subir-archivos.png" width="300">
</p>




<br>


Una vez lo hayáis subido, os saldrá un aviso importante: los datos no se guardan permanentemente, es decir que si por cualquier razón se reinicia el entorno de Collab, tendréis que volver a subir el archivo.


## Y ya estaría

Sé que ha parecido corto, pero no hace falta mucho más para usar Collab en esta guía para construir un GPT.

Lo básico que necesitamos es saber subir datos (Tarea 2) y ejecutar líneas de código (Tarea 1), con estas dos habilidades, ya estáis más que listos para continuar, solo necesitaréis saber cambiar de una CPU a una GPU en la lección final, pero eso lo veremos en dicha lección.

Nos vemos en la siguiente parte:
[Parte 2: Tokens](../partes/parte2.md)