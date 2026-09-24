#!/usr/bin/env python3

import os
import shutil
import tarfile
import datetime

import requests
import rioxarray as rxr
import cupy as cp


# ============================================================
# Configuration
# ============================================================

AEMET_URL = (
    "https://www.aemet.es/es/api-eltiempo/modelos/download/harmonie/PB"
)

BASE_DIR = os.getcwd()

HARMONIE_ARCHIVE = os.path.join(
    BASE_DIR,
    "harmonie.tar.gz"
)

HARMONIE_DIR = os.path.join(
    BASE_DIR,
    "harmonie_data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "prediccion-aemet-harmonie-arome",
    "tiff"
)

# Spain bounding box
LAT_MIN = 34.5
LAT_MAX = 44.3
LON_MIN = -9.65
LON_MAX = 4.5

# Number of forecast hours
FORECAST_HOURS = 48

# GPU palette chunk size.
#
# 8  = lower GPU memory usage
# 16 = good compromise
# 32 = fastest for your current palettes, but uses more VRAM
#
# You can also use None to process the whole palette at once.
PALETTE_CHUNK_SIZE = 16

# AEMET HARMONIE-AROME product codes
# ["_11.tif","_32.tif","_61_1HH.tif","_61_3HH.tif","_61_6HH.tif","_71.tif","_207.tif","_207_3HH.tif","_228.tif","_228_3HH.tif"]
# [temperatura, viento, lluvia1h, lluvia3h, lluvia6h, nubosidad, rayos,rayos3h, rachas, rachas3h]

D_CODES = [
    "_11.tif",
    "_61_1HH.tif",
]
#D_CODES =["_11.tif","_32.tif","_61_1HH.tif","_61_3HH.tif","_61_6HH.tif","_71.tif","_207.tif","_207_3HH.tif","_228.tif","_228_3HH.tif"]


# ============================================================
# GPU information
# ============================================================

def print_gpu_info():
    """Print CUDA GPU information."""

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)

        name = props["name"]

        if isinstance(name, bytes):
            name = name.decode()

        free_memory, total_memory = device.mem_info

        print()
        print("CUDA GPU:")
        print(f"  Device: {name}")
        print(
            f"  VRAM:   "
            f"{total_memory / (1024 ** 3):.2f} GB"
        )
        print(
            f"  Free:   "
            f"{free_memory / (1024 ** 3):.2f} GB"
        )
        print()

    except Exception as exc:
        raise RuntimeError(
            "CuPy could not access a CUDA GPU. "
            "Check that CUDA and the matching CuPy package "
            "are installed correctly."
        ) from exc


# ============================================================
# Filesystem
# ============================================================

def prepare_directories():
    """Remove old data and create output directories."""

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if os.path.exists(HARMONIE_DIR):
        shutil.rmtree(HARMONIE_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HARMONIE_DIR, exist_ok=True)


# ============================================================
# Download
# ============================================================

def download_forecast():
    """Download the AEMET HARMONIE-AROME forecast."""

    print("Downloading HARMONIE-AROME forecast...")

    response = requests.get(
        AEMET_URL,
        timeout=300
    )

    response.raise_for_status()

    with open(HARMONIE_ARCHIVE, "wb") as file:
        file.write(response.content)

    size_mb = (
        os.path.getsize(HARMONIE_ARCHIVE)
        / (1024 ** 2)
    )

    print(
        f"Downloaded {size_mb:.1f} MB"
    )


# ============================================================
# Extraction
# ============================================================

def extract_forecast():
    """Extract the downloaded forecast archive."""

    if not os.path.exists(HARMONIE_ARCHIVE):
        raise FileNotFoundError(
            f"Missing archive: {HARMONIE_ARCHIVE}"
        )

    print("Extracting HARMONIE-AROME data...")

    with tarfile.open(
        HARMONIE_ARCHIVE,
        "r:gz"
    ) as tar:
        tar.extractall(HARMONIE_DIR)

    print("Extraction complete.")


# ============================================================
# Colour palettes
# ============================================================

def get_palette(data_code):
    """
    Return the AEMET RGB palette and physical values.
    """

    if data_code == "_11.tif":

        return {
            "R": [
                122, 150, 181, 209, 138, 166, 196, 224,
                255, 26, 26, 28, 28, 31, 0, 10,
                20, 33, 102, 204, 255, 255, 255, 255,
                255, 255, 255, 232, 209, 178, 161, 138
            ],
            "G": [
                56, 48, 41, 33, 43, 97, 148, 201,
                255, 26, 54, 84, 112, 143, 237, 217,
                196, 178, 255, 255, 255, 222, 191,
                158, 128, 0, 51, 51, 51, 51, 54, 54
            ],
            "B": [
                140, 140, 140, 143, 227, 232, 240, 247,
                255, 112, 148, 184, 219, 255, 237, 214,
                191, 171, 102, 0, 0, 0, 0, 0,
                0, 0, 178, 145, 112, 79, 46, 15
            ],
            "v": [
                -27.5, -22.5, -17.5, -12.5,
                -9, -7, -5, -3, -1, 1, 3, 5,
                7, 9, 11, 13, 15, 17, 19, 21,
                23, 25, 27, 29, 31, 33, 35,
                37, 39, 41, 43, 44
            ]
        }

    elif data_code == "_32.tif":

        return {
            "R": [
                216, 175, 102, 80, 168, 217, 255,
                255, 255, 255, 255, 153, 153
            ],
            "G": [
                250, 249, 246, 249, 244, 242, 201,
                138, 105, 89, 120, 102, 51
            ],
            "B": [
                248, 246, 236, 183, 1, 0, 11,
                23, 31, 89, 212, 255, 204
            ],
            "v": [
                5, 15, 25, 35, 45, 55, 65,
                75, 85, 95, 105, 115, 120
            ]
        }

    elif data_code in (
        "_61_1HH.tif",
        "_61_3HH.tif",
        "_61_6HH.tif"
    ):

        return {
            "R": [
                19, 176, 51, 0, 0, 0, 128, 191,
                255, 255, 255, 255, 204, 219, 236
            ],
            "G": [
                49, 224, 245, 204, 178, 153, 204,
                230, 255, 186, 122, 61, 84, 141, 200
            ],
            "B": [
                52, 230, 222, 128, 64, 0, 0, 0,
                0, 15, 8, 3, 83, 140, 200
            ],
            "v": [
                0, 0.75, 1.5, 3.5, 7.5, 15,
                25, 35, 50, 70, 90, 110,
                150, 215, 275
            ]
        }

    elif data_code == "_71.tif":

        return {
            "R": [
                255, 217, 229, 196, 160,
                114, 76, 33, 12
            ],
            "G": [
                255, 217, 235, 214, 196,
                169, 141, 115, 41
            ],
            "B": [
                255, 219, 250, 229, 211,
                195, 178, 184, 74
            ],
            "v": [
                15, 25, 35, 45, 55,
                65, 75, 85, 90
            ]
        }

    elif data_code in (
        "_207.tif",
        "_207_3HH.tif"
    ):

        return {
            "R": [
                53, 47, 41, 35, 63, 147,
                233, 232, 229, 224, 191
            ],
            "G": [
                151, 231, 247, 244, 241, 238,
                235, 144, 48, 0, 0
            ],
            "B": [
                253, 250, 180, 90, 30, 24,
                19, 14, 9, 147, 61
            ],
            "v": [
                0.0105, 0.03, 0.05, 0.07,
                0.09, 0.11, 0.13, 0.15,
                0.17, 0.19, 0.2
            ]
        }

    elif data_code in (
        "_228.tif",
        "_228_3HH.tif"
    ):

        return {
            "R": [
                217, 176, 102, 79, 168, 217, 255,
                255, 255, 255, 255, 153, 153,
                171, 190
            ],
            "G": [
                248, 250, 247, 250, 245, 242, 201,
                138, 105, 89, 120, 102, 51,
                26, 0
            ],
            "B": [
                248, 247, 237, 184, 0, 0, 10,
                23, 31, 89, 212, 255, 204,
                133, 61
            ],
            "v": [
                5, 15, 25, 35, 45, 55, 65,
                75, 85, 95, 105, 115, 125,
                135, 140
            ]
        }

    else:
        raise ValueError(
            f"No palette defined for {data_code}"
        )


# ============================================================
# GPU palette
# ============================================================

def prepare_gpu_palette(palette):
    """
    Copy the colour palette to GPU memory.
    """

    return {
        "R": cp.asarray(
            palette["R"],
            dtype=cp.int16
        ),
        "G": cp.asarray(
            palette["G"],
            dtype=cp.int16
        ),
        "B": cp.asarray(
            palette["B"],
            dtype=cp.int16
        ),
        "v": cp.asarray(
            palette["v"],
            dtype=cp.float32
        )
    }


# ============================================================
# GPU raster conversion
# ============================================================

def convert_raster_gpu(
    hxr,
    gpu_palette,
    chunk_size=16
):
    """
    Convert the RGB raster to physical values on the GPU.

    For every pixel:

        distance =
            abs(R - palette_R)
          + abs(G - palette_G)
          + abs(B - palette_B)

    The palette entry with the smallest distance is selected.

    This replaces the original pandas DataFrame processing and
    Python per-pixel loop.
    """

    # --------------------------------------------------------
    # Read source raster into CPU memory
    # --------------------------------------------------------

    data = hxr.values

    if data.ndim != 3:
        raise ValueError(
            f"Expected raster with 3 dimensions, "
            f"got {data.shape}"
        )

    if data.shape[0] < 4:
        raise ValueError(
            f"Expected at least 4 bands, "
            f"got {data.shape[0]}"
        )

    # --------------------------------------------------------
    # Transfer raster to GPU
    # --------------------------------------------------------

    rgb = cp.asarray(
        data[:3],
        dtype=cp.int16
    )

    original_value = cp.asarray(
        data[3],
        dtype=cp.float32
    )

    R = rgb[0]
    G = rgb[1]
    B = rgb[2]

    # Convert once instead of doing it for every palette chunk.
    R32 = R.astype(cp.int32)
    G32 = G.astype(cp.int32)
    B32 = B.astype(cp.int32)

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------
    #
    # Original code:
    #
    # Tband = fourth band
    # Nband = fourth band + 999
    # Vband = fourth band
    #
    # We preserve that behavior.

    best_distance = (
        original_value.astype(cp.int32) + 999
    )

    best_value = original_value

    scale_R = gpu_palette["R"]
    scale_G = gpu_palette["G"]
    scale_B = gpu_palette["B"]
    scale_V = gpu_palette["v"]

    palette_size = len(scale_V)

    if chunk_size is None:
        chunk_size = palette_size

    # --------------------------------------------------------
    # Compare against palette
    # --------------------------------------------------------

    for start in range(
        0,
        palette_size,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            palette_size
        )

        sr = scale_R[start:end]
        sg = scale_G[start:end]
        sb = scale_B[start:end]
        sv = scale_V[start:end]

        # ----------------------------------------------------
        # Vectorized GPU RGB distance
        # ----------------------------------------------------

        distance = (
            cp.abs(
                R32[..., None]
                - sr[None, None, :]
            )
            +
            cp.abs(
                G32[..., None]
                - sg[None, None, :]
            )
            +
            cp.abs(
                B32[..., None]
                - sb[None, None, :]
            )
        )

        # ----------------------------------------------------
        # Find closest palette colour
        # ----------------------------------------------------

        local_index = cp.argmin(
            distance,
            axis=2
        )

        local_distance = cp.take_along_axis(
            distance,
            local_index[..., None],
            axis=2
        )[..., 0]

        local_value = sv[local_index]

        # ----------------------------------------------------
        # Update best result
        # ----------------------------------------------------
        #
        # '<' is intentional.
        #
        # The original Python code only updated when:
        #
        #     new_distance < old_distance
        #
        # Therefore the first palette entry wins ties.

        mask = (
            local_distance
            < best_distance
        )

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

        # Release temporary arrays.
        del distance
        del local_index
        del local_distance
        del local_value
        del mask

    # --------------------------------------------------------
    # Wait for GPU operations
    # --------------------------------------------------------

    cp.cuda.Stream.null.synchronize()

    # --------------------------------------------------------
    # Copy final result GPU -> CPU
    # --------------------------------------------------------

    result = cp.asnumpy(
        best_value
    )

    # --------------------------------------------------------
    # Release GPU memory
    # --------------------------------------------------------

    del rgb
    del R32
    del G32
    del B32
    del original_value
    del best_distance
    del best_value

    return result


# ============================================================
# Process one TIFF
# ============================================================

def process_file(
    input_path,
    output_path,
    gpu_palette,
    chunk_size
):
    """
    Read, crop, convert on GPU and write one GeoTIFF.
    """

    print()
    print(
        f"Processing: "
        f"{os.path.basename(input_path)}"
    )

    # --------------------------------------------------------
    # Open raster
    # --------------------------------------------------------

    hxr = rxr.open_rasterio(
        input_path
    )

    # --------------------------------------------------------
    # Crop to Spain
    # --------------------------------------------------------

    hxr = hxr.sel(
        y=slice(
            LAT_MAX,
            LAT_MIN
        ),
        x=slice(
            LON_MIN,
            LON_MAX
        )
    )

    print(
        f"Raster shape: "
        f"{hxr.shape[1]} x {hxr.shape[2]}"
    )

    # --------------------------------------------------------
    # GPU conversion
    # --------------------------------------------------------

    result = convert_raster_gpu(
        hxr,
        gpu_palette,
        chunk_size=chunk_size
    )

    # --------------------------------------------------------
    # Use fourth band as output template
    # --------------------------------------------------------

    band = hxr.isel(
        band=3
    ).copy(
        deep=True
    )

    # Replace values with GPU result.
    band.values = result

    # --------------------------------------------------------
    # Write GeoTIFF
    # --------------------------------------------------------

    band.rio.to_raster(
        output_path,
        dtype="float32"
    )

    print(
        f"Written: {output_path}"
    )

    # --------------------------------------------------------
    # Release CPU/GPU memory
    # --------------------------------------------------------

    del hxr
    del band
    del result

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("==============================================")
    print(" AEMET HARMONIE-AROME")
    print(" CuPy / CUDA GPU processor")
    print("==============================================")

    # --------------------------------------------------------
    # Check GPU
    # --------------------------------------------------------

    print_gpu_info()

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    prepare_directories()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    download_forecast()

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    extract_forecast()

    # --------------------------------------------------------
    # Locate extracted files
    # --------------------------------------------------------

    os.chdir(
        HARMONIE_DIR
    )

    files = sorted(
        os.listdir()
    )

    if not files:
        raise RuntimeError(
            "No files found in harmonie_data"
        )

    # --------------------------------------------------------
    # Get initial forecast time
    # --------------------------------------------------------

    idate = files[0][5:24]

    idate = datetime.datetime.strptime(
        idate,
        "%Y-%m-%dT%H:%M:%S"
    )

    print()
    print(
        f"Forecast initial time: {idate}"
    )

    # --------------------------------------------------------
    # Process each product
    # --------------------------------------------------------

    for data_code in D_CODES:

        print()
        print("==============================================")
        print(
            f"Product: {data_code}"
        )
        print("==============================================")

        # ----------------------------------------------------
        # Load palette
        # ----------------------------------------------------

        palette = get_palette(
            data_code
        )

        # ----------------------------------------------------
        # Send palette to GPU once
        # ----------------------------------------------------

        gpu_palette = prepare_gpu_palette(
            palette
        )

        print(
            f"Palette entries: "
            f"{len(palette['v'])}"
        )

        # ----------------------------------------------------
        # Process forecast hours
        # ----------------------------------------------------

        for hour in range(
            FORECAST_HOURS
        ):

            d_time = (
                idate
                + datetime.timedelta(
                    hours=hour
                )
            )

            d_file = (
                "down_"
                + d_time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
                + "+00:00"
                + data_code
            )

            input_path = os.path.join(
                HARMONIE_DIR,
                d_file
            )

            # ------------------------------------------------
            # Check source file
            # ------------------------------------------------

            if not os.path.exists(
                input_path
            ):
                print(
                    f"{d_file} ERR"
                )
                continue

            print(
                f"{d_file} OK"
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            output_path = os.path.join(
                OUTPUT_DIR,
                d_file + "f"
            )

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            process_file(
                input_path,
                output_path,
                gpu_palette,
                PALETTE_CHUNK_SIZE
            )

        # ----------------------------------------------------
        # Release palette
        # ----------------------------------------------------

        del gpu_palette

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    print()
    print("==============================================")
    print(
        "AEMET HARMONIE-AROME forecast "
        "successfully processed!"
    )
    print("==============================================")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()