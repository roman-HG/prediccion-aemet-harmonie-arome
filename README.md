# prediccion-aemet-harmonie-arome
Aemet Harmonie-Arome reinterpretado en GeoTIFF

https://www.aemet.es/es/eltiempo/prediccion/modelosnumericos/harmonie_arome

Los datos públicos de Aemet solo facilitan un .tif en RGB que entorpecen cualquier interpretación de los valores.

El script **arome2geoTiff.py** realiza ingeniería inversa, traduciendo el valor de cada píxel según la escala de color facilitada, y transforma los archivos a GeoTiff (.tiff). **Está reciclado de otro proceso que utilizo para generar series de datos de predicción de Aemet, así que seguramente se hayan colado algunas líneas que sobran.*

La carpeta /tiff contiene archivos de ejemplo de temperatura ("_11.tiff") y precipitación en 1 hora ("_61_1HH.tiff"), que se actualizan 2 veces al día con los archivos más recientes.

Dependencias:
* rioxarray
* pandas
