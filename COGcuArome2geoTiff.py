#!/usr/bin/env python3

import os
import shutil
import tarfile
import datetime

import numpy as np
import requests
import rioxarray as rxr
import cupy as cp

from rioxarray.merge import merge_arrays


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.getcwd()

PB_DIR = os.path.join(BASE_DIR, "harmonie_data_pb")
CAN_DIR = os.path.join(BASE_DIR, "harmonie_data_can")

PB_ARCHIVE = os.path.join(BASE_DIR, "harmonie_pb.tar.gz")
CAN_ARCHIVE = os.path.join(BASE_DIR, "harmonie_can.tar.gz")

TIFF_DIR = os.path.join(
    BASE_DIR,
    "prediccion-aemet-harmonie-arome",
    "tiff"
)

PB_URL = (
    "https://www.aemet.es/es/api-eltiempo/"
    "modelos/download/harmonie/PB"
)

CAN_URL = (
    "https://www.aemet.es/es/api-eltiempo/"
    "modelos/download/harmonie/CAN"
)

LAT_MIN = 27.0
LAT_MAX = 44.3
LON_MIN = -18.5
LON_MAX = 4.5

N_FORECAST_HOURS = 48

NODATA = -999.0

PALETTE_CHUNK_SIZE = 16

D_CODES = [
    "_11.tif",
    "_32.tif",
    "_61_1HH.tif",
    "_61_3HH.tif",
    "_61_6HH.tif",
    "_71.tif",
    "_207.tif",
    "_207_3HH.tif",
    "_228.tif",
    "_228_3HH.tif",
]


# ============================================================
# AEMET COLOUR SCALES
# ============================================================

SCALES = {
    "_11.tif": {
        "R": [122,150,181,209,138,166,196,224,255,26,26,28,28,31,0,10,20,33,102,204,255,255,255,255,255,255,255,232,209,178,161,138],
        "G": [56,48,41,33,43,97,148,201,255,26,54,84,112,143,237,217,196,178,255,255,255,222,191,158,128,0,51,51,51,51,54,54],
        "B": [140,140,140,143,227,232,240,247,255,112,148,184,219,255,237,214,191,171,102,0,0,0,0,0,0,0,178,145,112,79,46,15],
        "v": [-27.5,-22.5,-17.5,-12.5,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,44],
    },

    "_32.tif": {
        "R": [216,175,102,80,168,217,255,255,255,255,255,153,153],
        "G": [250,249,246,249,244,242,201,138,105,89,120,102,51],
        "B": [248,246,236,183,1,0,11,23,31,89,212,255,204],
        "v": [5,15,25,35,45,55,65,75,85,95,105,115,120],
    },

    "_61_1HH.tif": {
        "R": [19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G": [49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B": [52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v": [0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275],
    },

    "_61_3HH.tif": {
        "R": [19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G": [49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B": [52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v": [0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275],
    },

    "_61_6HH.tif": {
        "R": [19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G": [49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B": [52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v": [0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275],
    },

    "_71.tif": {
        "R": [255,217,229,196,160,114,76,33,12],
        "G": [255,217,235,214,196,169,141,115,41],
        "B": [255,219,250,229,211,195,178,184,74],
        "v": [15,25,35,45,55,65,75,85,90],
    },

    "_207.tif": {
        "R": [53,47,41,35,63,147,233,232,229,224,191],
        "G": [151,231,247,244,241,238,235,144,48,0,0],
        "B": [253,250,180,90,30,24,19,14,9,147,61],
        "v": [0.0105,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.2],
    },

    "_207_3HH.tif": {
        "R": [53,47,41,35,63,147,233,232,229,224,191],
        "G": [151,231,247,244,241,238,235,144,48,0,0],
        "B": [253,250,180,90,30,24,19,14,9,147,61],
        "v": [0.0105,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.2],
    },

    "_228.tif": {
        "R": [217,176,102,79,168,217,255,255,255,255,255,153,153,171,190],
        "G": [248,250,247,250,245,242,201,138,105,89,120,102,51,26,0],
        "B": [248,247,237,184,0,0,10,23,31,89,212,255,204,133,61],
        "v": [5,15,25,35,45,55,65,75,85,95,105,115,125,135,140],
    },

    "_228_3HH.tif": {
        "R": [217,176,102,79,168,217,255,255,255,255,255,153,153,171,190],
        "G": [248,250,247,250,245,242,201,138,105,89,120,102,51,26,0],
        "B": [248,247,237,184,0,0,10,23,31,89,212,255,204,133,61],
        "v": [5,15,25,35,45,55,65,75,85,95,105,115,125,135,140],
    },
}


# ============================================================
# GPU
# ============================================================

def print_gpu_info():
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)

        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()

        free_memory, total_memory = device.mem_info

        print(f"\nCUDA GPU: {name}")
        print(f"VRAM: {total_memory / 1024**3:.2f} GB")
        print(f"Free: {free_memory / 1024**3:.2f} GB\n")

    except Exception as exc:
        raise RuntimeError(
            "CuPy no puede acceder a una GPU CUDA. "
            "Comprueba la instalación de CUDA y CuPy."
        ) from exc


# ============================================================
# DIRECTORIES AND DOWNLOAD
# ============================================================

def prepare_directories():
    for directory in (PB_DIR, CAN_DIR):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(TIFF_DIR):
        shutil.rmtree(TIFF_DIR)

    os.makedirs(TIFF_DIR, exist_ok=True)


def download_file(url, output_file):
    print(f"Descargando: {url}")

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    with open(output_file, "wb") as file:
        file.write(response.content)

    print(
        f"Descargado: {output_file} "
        f"({os.path.getsize(output_file) / 1024**2:.1f} MB)"
    )


def extract_archive(archive, directory):
    print(f"Extrayendo: {archive}")

    with tarfile.open(archive, "r:gz") as tar:
        # Evitar extraer rutas que escapen del directorio destino.
        base = os.path.realpath(directory)

        for member in tar.getmembers():
            target = os.path.realpath(
                os.path.join(directory, member.name)
            )

            if os.path.commonpath([base, target]) != base:
                raise RuntimeError(
                    f"Ruta no válida en el archivo: {member.name}"
                )

        tar.extractall(directory)

    print("Extracción completada.")


# ============================================================
# FILE LOOKUP
# ============================================================

def get_files(directory):
    result = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(".tif"):
                result.append(os.path.join(root, filename))

    return sorted(result)


def build_file_lookup(directory):
    return {
        os.path.basename(path): path
        for path in get_files(directory)
    }


def build_filename(idate, hour, d_code):
    forecast_time = idate + datetime.timedelta(hours=hour)

    return (
        "down_"
        + forecast_time.strftime("%Y-%m-%dT%H:%M:%S")
        + "+00:00"
        + d_code
    )


# ============================================================
# RASTER CLEANING: FIX nodata NaN / INTEGER ERROR
# ============================================================

def open_raster_clean(path):
    """
    Open raster, normalize its dtype and nodata before merging.
    This prevents NaN nodata values being cast to integer types.
    """
    raster = rxr.open_rasterio(path, masked=True)

    try:
        raster = raster.astype(np.float32)

        # Clear potentially incompatible inherited encoding.
        for key in ("dtype", "_FillValue", "missing_value"):
            raster.encoding.pop(key, None)

        # Replace NaN and infinities.
        raster = raster.where(np.isfinite(raster), NODATA)
        raster = raster.fillna(NODATA)

        raster = raster.rio.write_nodata(
            NODATA,
            encoded=False,
            inplace=False
        )

        return raster

    except Exception:
        raster.close()
        raise


def merge_forecast_files(pb_file, can_file):
    print(f"  PB:  {pb_file}")
    print(f"  CAN: {can_file}")

    pb = open_raster_clean(pb_file)
    can = open_raster_clean(can_file)

    try:
        if pb.sizes.get("band") != can.sizes.get("band"):
            raise ValueError(
                "PB y CAN tienen distinto número de bandas: "
                f"{pb.sizes.get('band')} y {can.sizes.get('band')}"
            )

        merged = merge_arrays(
            [pb, can],
            nodata=NODATA
        )

        # Normalize merged output too.
        merged = merged.astype(np.float32)

        for key in ("dtype", "_FillValue", "missing_value"):
            merged.encoding.pop(key, None)

        merged = merged.where(np.isfinite(merged), NODATA)
        merged = merged.fillna(NODATA)

        merged = merged.rio.write_nodata(
            NODATA,
            encoded=False,
            inplace=False
        )

        return merged

    finally:
        pb.close()
        can.close()


# ============================================================
# GPU PALETTE
# ============================================================

def prepare_gpu_palette(scale):
    rgb = np.column_stack(
        (scale["R"], scale["G"], scale["B"])
    ).astype(np.int16)

    values = np.asarray(scale["v"], dtype=np.float32)

    return {
        "R": cp.asarray(rgb[:, 0], dtype=cp.int16),
        "G": cp.asarray(rgb[:, 1], dtype=cp.int16),
        "B": cp.asarray(rgb[:, 2], dtype=cp.int16),
        "v": cp.asarray(values, dtype=cp.float32),
    }


# ============================================================
# RGB CLASSIFICATION ON GPU
# ============================================================

def classify_raster_gpu(raster, gpu_palette, chunk_size=16):
    """
    Classify RGB pixels using CuPy.

    Pixels with invalid RGB data are assigned NODATA (-999).
    """
    data = np.asarray(raster.values)

    if data.ndim != 3 or data.shape[0] < 4:
        raise ValueError(
            f"Se esperaban al menos 4 bandas; forma recibida: {data.shape}"
        )

    rgb_source = data[:3]

    # Invalid RGB pixels must remain nodata.
    invalid_pixels = ~np.all(
        np.isfinite(rgb_source),
        axis=0
    )

    # Transfer RGB to GPU. int32 avoids arithmetic overflow.
    rgb = cp.asarray(
        np.nan_to_num(
            rgb_source,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ),
        dtype=cp.int32
    )

    R, G, B = rgb[0], rgb[1], rgb[2]

    scale_R = gpu_palette["R"].astype(cp.int32)
    scale_G = gpu_palette["G"].astype(cp.int32)
    scale_B = gpu_palette["B"].astype(cp.int32)
    scale_values = gpu_palette["v"]

    palette_size = int(scale_values.size)

    if chunk_size is None or chunk_size <= 0:
        chunk_size = palette_size

    best_distance = cp.full(
        R.shape,
        np.iinfo(np.int32).max,
        dtype=cp.int32
    )

    best_value = cp.full(
        R.shape,
        NODATA,
        dtype=cp.float32
    )

    for start in range(0, palette_size, chunk_size):
        end = min(start + chunk_size, palette_size)

        sr = scale_R[start:end]
        sg = scale_G[start:end]
        sb = scale_B[start:end]
        sv = scale_values[start:end]

        distances = (
            cp.abs(R[..., None] - sr[None, None, :])
            + cp.abs(G[..., None] - sg[None, None, :])
            + cp.abs(B[..., None] - sb[None, None, :])
        )

        local_index = cp.argmin(distances, axis=2)

        local_distance = cp.take_along_axis(
            distances,
            local_index[..., None],
            axis=2
        )[..., 0]

        local_value = sv[local_index]

        # Strict less-than preserves the first palette colour on ties.
        mask = local_distance < best_distance

        best_distance = cp.where(
            mask,
            local_distance,
            best_distance
        )

        best_value = cp.where(
            mask,
            local_value,
            best_value
        )

        del distances, local_index, local_distance, local_value, mask

    cp.cuda.Stream.null.synchronize()

    result = cp.asnumpy(best_value).astype(np.float32)

    # Set invalid source pixels to nodata on CPU.
    result[invalid_pixels] = NODATA

    result = np.nan_to_num(
        result,
        nan=NODATA,
        posinf=NODATA,
        neginf=NODATA
    ).astype(np.float32)

    del rgb, R, G, B, best_distance, best_value
    cp.get_default_memory_pool().free_all_blocks()

    return result


# ============================================================
# CROP AND SAVE COG
# ============================================================

def crop_raster(raster):
    """
    Crop to Spain and Canary Islands.
    Handles both ascending and descending coordinate axes.
    """
    x = raster.x.values
    y = raster.y.values

    if x.size == 0 or y.size == 0:
        raise ValueError("El raster no contiene coordenadas.")

    if y[0] > y[-1]:
        y_slice = slice(LAT_MAX, LAT_MIN)
    else:
        y_slice = slice(LAT_MIN, LAT_MAX)

    if x[0] > x[-1]:
        x_slice = slice(LON_MAX, LON_MIN)
    else:
        x_slice = slice(LON_MIN, LON_MAX)

    cropped = raster.sel(y=y_slice, x=x_slice)

    if cropped.sizes.get("x", 0) == 0 or cropped.sizes.get("y", 0) == 0:
        raise ValueError(
            "El recorte geográfico no contiene píxeles. "
            "Comprueba el CRS y los límites geográficos."
        )

    return cropped


def save_cog(result, reference_raster, output_file):
    result = np.asarray(result, dtype=np.float32)

    result = np.nan_to_num(
        result,
        nan=NODATA,
        posinf=NODATA,
        neginf=NODATA
    ).astype(np.float32)

    # Use the fourth band as the geospatial template.
    output_band = reference_raster.isel(
        band=3
    ).copy(deep=True)

    output_band = output_band.astype(np.float32)

    if result.shape != output_band.shape:
        raise ValueError(
            f"Dimensiones incompatibles: resultado {result.shape}, "
            f"raster {output_band.shape}"
        )

    for key in ("dtype", "_FillValue", "missing_value"):
        output_band.encoding.pop(key, None)

    output_band.values = result

    output_band = output_band.rio.write_nodata(
        NODATA,
        encoded=False,
        inplace=False
    )

    output_band.rio.to_raster(
        output_file,
        driver="COG",
        dtype="float32",
        nodata=NODATA,
        compress="DEFLATE",
        predictor=3,
        BIGTIFF="IF_SAFER",
        OVERVIEWS="AUTO",
        OVERVIEW_RESAMPLING="NEAREST"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print(" AEMET HARMONIE-AROME — CuPy GPU — PB + CAN")
    print("=" * 65)

    print_gpu_info()
    prepare_directories()

    # Download both forecast domains.
    download_file(PB_URL, PB_ARCHIVE)
    download_file(CAN_URL, CAN_ARCHIVE)

    # Extract both archives.
    extract_archive(PB_ARCHIVE, PB_DIR)
    extract_archive(CAN_ARCHIVE, CAN_DIR)

    # Build file lookup dictionaries.
    pb_files = build_file_lookup(PB_DIR)
    can_files = build_file_lookup(CAN_DIR)

    print(f"\nPB TIFF files:  {len(pb_files)}")
    print(f"CAN TIFF files: {len(can_files)}")

    if not pb_files:
        raise RuntimeError("No se encontraron TIFF en los datos PB.")

    if not can_files:
        raise RuntimeError("No se encontraron TIFF en los datos CAN.")

    # Determine forecast initialization time.
    first_file = sorted(pb_files.keys())[0]

    idate = datetime.datetime.strptime(
        first_file[5:24],
        "%Y-%m-%dT%H:%M:%S"
    )

    print(f"\nForecast initialization: {idate}")

    for d_code in D_CODES:
        print("\n" + "=" * 65)
        print(f"PROCESSING {d_code}")
        print("=" * 65)

        scale = SCALES[d_code]
        gpu_palette = prepare_gpu_palette(scale)

        for hour in range(N_FORECAST_HOURS):
            filename = build_filename(idate, hour, d_code)

            pb_file = pb_files.get(filename)
            can_file = can_files.get(filename)

            print(f"\nHour {hour:02d}: {filename}")

            if pb_file is None:
                print("  PB file missing; skipping.")
                continue

            if can_file is None:
                print("  CAN file missing; skipping.")
                continue

            merged = None

            try:
                # Open and normalize nodata before merging.
                merged = merge_forecast_files(
                    pb_file,
                    can_file
                )

                # Crop the combined raster.
                merged = crop_raster(merged)

                print(
                    f"  Raster size: "
                    f"{merged.sizes['y']} x {merged.sizes['x']}"
                )

                # Classify RGB on the GPU.
                result = classify_raster_gpu(
                    merged,
                    gpu_palette,
                    chunk_size=PALETTE_CHUNK_SIZE
                )

                output_file = os.path.join(
                    TIFF_DIR,
                    filename + "f"
                )

                # Save as COG float32 with nodata=-999.
                save_cog(
                    result,
                    merged,
                    output_file
                )

                print(f"  Saved: {output_file}")

                del result

            except Exception as exc:
                print(f"  ERROR processing {filename}: {exc}")
                raise

            finally:
                if merged is not None:
                    try:
                        merged.close()
                    except Exception:
                        pass

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

        del gpu_palette
        cp.get_default_memory_pool().free_all_blocks()

    print("\n" + "=" * 65)
    print("Processing completed.")
    print(f"Output directory: {TIFF_DIR}")
    print(f"Nodata value: {NODATA}")
    print("=" * 65)


if __name__ == "__main__":
    main()