# prediccion-aemet-harmonie-arome v1
Aemet Harmonie-Arome reinterpretado en GeoTIFF

https://www.aemet.es/es/eltiempo/prediccion/modelosnumericos/harmonie_arome

Los datos públicos de Aemet solo facilitan un .tif en RGB que entorpecen cualquier interpretación de los valores.

El script **arome2geoTiff.py** realiza ingeniería inversa, traduciendo el valor de cada píxel según la escala de color facilitada, y transforma los archivos a GeoTiff (.tiff). **Está reciclado de otro proceso que utilizo para generar series de datos de predicción de Aemet, así que seguramente se hayan colado algunas líneas que sobran.*

Este script se ejecuta en un servidor propio que actualiza la carpeta /tiff 2 veces al día con los archivos más recientes. La fecha de actualización se mostrará en el commit de la carpeta, así como en el nombre de los ficheros, que mantienen su nomenclatura original, con extensión **.tiff** .

Dependencias:
* rioxarray
* pandas
