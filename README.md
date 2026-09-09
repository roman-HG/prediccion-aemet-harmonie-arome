# prediccion-aemet-harmonie-arome
Aemet Harmonie-Arome reinterpretado en GeoTIFF

https://www.aemet.es/es/eltiempo/prediccion/modelosnumericos/harmonie_arome

Los datos públicos de Aemet solo facilitan un .tif en RGB que entorpecen cualquier interpretación de los valores.

El script **arome2geoTiff.py** realiza ingeniería inversa, traduciendo el valor de cada píxel según la escala de color facilitada, y transforma los archivos a GeoTiff (.tiff).
* **d_code[]** permite definir las variables a procesar.
* **lat_min, lat_max** y **lon_min, lon_max** definen la extensión a procesar. (Toda España consume muchos recursos)


La carpeta /tiff contiene archivos de ejemplo de temperatura ("_11.tiff") y precipitación en 1 hora ("_61_1HH.tiff"), que se actualizan 2 veces al día con los archivos más recientes.

Dependencias de python:
* rioxarray
* pandas

**index.html** es un visor que utiliza Leaflet.js para previsualizar las capas actualmente descargadas en la carpeta /tiff
