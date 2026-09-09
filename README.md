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

**index.html** es un visor que utiliza Leaflet.js para previsualizar las capas actualmente descargadas en la carpeta /tiff. Está publicado mediante Github Pages a través del siguiente enlace:

https://roman-hg.github.io/prediccion-aemet-harmonie-arome/

_Nota:_
Los valores mostrados por Aemet son UTC+2 (Madrid), según el cambio de horario verano-invierno.
Los valores mostrados en la carpeta de descarga son UTC+0.

<img width="300" height="300" alt="aemet18h" src="https://github.com/user-attachments/assets/3e921b63-624b-43dd-9039-a30775f30e6c" />        <img width="300" height="300" alt="descarga18h" src="https://github.com/user-attachments/assets/9ea64813-a554-496f-b974-869d6bad8cdc" />


