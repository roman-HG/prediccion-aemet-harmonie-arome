# prediccion-aemet-harmonie-arome
Aemet Harmonie-Arome reinterpretado en Cloud Optimized GeoTIFF

https://www.aemet.es/es/eltiempo/prediccion/modelosnumericos/harmonie_arome

Los datos públicos de Aemet solo facilitan un .tif en RGB que entorpecen cualquier interpretación de los valores.

Este repositorio transforma los archivos originales mediante distintas técnicas, buscando obtener un valor único por píxel que sea directamente interpretable por visores web y de escritorio.

## Scripts
### arome2geotiff.py

Script original que realiza ingeniería inversa, traduciendo el valor de cada píxel según la escala de color facilitada, y transforma los archivos a GeoTiff (.tiff).
* **d_code[]** permite definir las variables a procesar.
* **lat_min, lat_max** y **lon_min, lon_max** definen la extensión a procesar. (Toda España consume muchos recursos)

### cuArome2geoTiff.py

Versión optimizada del anterior script que emplea la librería CuPy para acelerar los cálculos de los rásters a través de la GPU mediante CUDA.

### piArome2geoTiff.py

Versión revisada con IA del script original para correr en pequeños dispositivos como Raspberry Pi. Optimiza los cálculos de arrays con NumPy sin aceleración de GPU. Más lento que la versión CUDA, pero aceptable.

### COGarome2geoTIFF.py

Revisión del script **piArome2geoTiff.py** con salida a formato COG (Cloud Optimized GeoTIFF) para facilitar la representación en mapas web.

## Archivos TIFF

La carpeta **/tiff** contiene los últimos archivos de predicción de Aemet, ya procesados con una frecuencia de 4 veces al día, y en formato Cloud Optimized GeoTIFF.

Se incluyen todo los intervalos horarios y todas la variables que se descargan libremente desde la página web de Aemet.

## Visor

Se ha habilitado un visor mediante Github Pages que permite previsualizar e incluso descargar los archivos de la carpeta **/tiff**. Se puede acceder al visor mediante el siguiente enlace:

<a href="https://roman-hg.github.io/prediccion-aemet-harmonie-arome/" target="_blank">https://roman-hg.github.io/prediccion-aemet-harmonie-arome/</a>

Github Pages renderiza el contenido del archivo **index.html**, ejemplo de uso de Leaflet.js para visualizar mapas web y capas geoespaciales complejas.


_Nota:
Los valores mostrados por Aemet son UTC+2 (Madrid), según el cambio de horario verano-invierno.
Los valores mostrados en la carpeta de descarga son UTC+0._

<img width="300" height="300" alt="aemet18h" src="https://github.com/user-attachments/assets/3e921b63-624b-43dd-9039-a30775f30e6c" />        <img width="300" height="300" alt="descarga18h" src="https://github.com/user-attachments/assets/9ea64813-a554-496f-b974-869d6bad8cdc" />


**Dependencias de python:**
* rioxarray
* pandas
* _cupy_ (en caso de usar el script cuArome2geoTiff.py, que emplea CUDA y requiere una GPU compatible)
* _gdal_ (en caso de usar el script COGarome2geoTIFF.py y producir Cloud Optimized GeoTIFFs)
