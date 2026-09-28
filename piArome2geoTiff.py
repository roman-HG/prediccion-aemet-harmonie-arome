# python3 -m pip install rioxarray rasterio
# conda activate pyNOAAenv

import os
import shutil
import tarfile
import datetime

import numpy as np
import requests
import rioxarray as rxr

from rioxarray.merge import merge_arrays


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.getcwd()

PB_DIR = os.path.join(BASE_DIR, "harmonie_data_pb")
CAN_DIR = os.path.join(BASE_DIR, "harmonie_data_can")

TIFF_DIR = os.path.join(
    BASE_DIR,
    "prediccion-aemet-harmonie-arome",
    "tiff"
)

PB_ARCHIVE = os.path.join(
    BASE_DIR,
    "harmonie_pb.tar.gz"
)

CAN_ARCHIVE = os.path.join(
    BASE_DIR,
    "harmonie_can.tar.gz"
)

PB_URL = (
    "https://www.aemet.es/es/api-eltiempo/"
    "modelos/download/harmonie/PB"
)

CAN_URL = (
    "https://www.aemet.es/es/api-eltiempo/"
    "modelos/download/harmonie/CAN"
)


# ------------------------------------------------------------
# Combined geographical area
# ------------------------------------------------------------

# Spain + Canary Islands
LAT_MIN = 27.0
LAT_MAX = 44.3

LON_MIN = -18.5
LON_MAX = 4.5


N_FORECAST_HOURS = 48


# Products
'''
D_CODES = [
    "_11.tif",      # temperatura
    "_32.tif",      # viento
    "_61_1HH.tif",  # lluvia 1h
    "_61_3HH.tif",  # lluvia 3h
    "_61_6HH.tif",  # lluvia 6h
    "_71.tif",      # nubosidad
    "_207.tif",     # rayos
    "_207_3HH.tif", # rayos 3h
    "_228.tif",     # rachas
    "_228_3HH.tif"  # rachas 3h
]
'''
D_CODES = [
    "_11.tif",
    "_61_1HH.tif",
]


# ============================================================
# AEMET colour scales
# ============================================================
SCALES = {

    "_11.tif": {
        "R":[122,150,181,209,138,166,196,224,255,26,26,28,28,31,0,10,20,33,102,204,255,255,255,255,255,255,255,232,209,178,161,138],
        "G":[56,48,41,33,43,97,148,201,255,26,54,84,112,143,237,217,196,178,255,255,255,222,191,158,128,0,51,51,51,51,54,54],
        "B":[140,140,140,143,227,232,240,247,255,112,148,184,219,255,237,214,191,171,102,0,0,0,0,0,0,0,178,145,112,79,46,15],
        "v":[-27.5,-22.5,-17.5,-12.5,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,44]
    },

    "_32.tif": {
        "R":[216,175,102,80,168,217,255,255,255,255,255,153,153],
        "G":[250,249,246,249,244,242,201,138,105,89,120,102,51],
        "B":[248,246,236,183,1,0,11,23,31,89,212,255,204],
        "v":[5,15,25,35,45,55,65,75,85,95,105,115,120]
    },

    "_61_1HH.tif": {
        "R":[19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G":[49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B":[52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v":[0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275]
    },

    "_61_3HH.tif": {
        "R":[19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G":[49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B":[52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v":[0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275]
    },

    "_61_6HH.tif": {
        "R":[19,176,51,0,0,0,128,191,255,255,255,255,204,219,236],
        "G":[49,224,245,204,178,153,204,230,255,186,122,61,84,141,200],
        "B":[52,230,222,128,64,0,0,0,0,15,8,3,83,140,200],
        "v":[0,0.75,1.5,3.5,7.5,15,25,35,50,70,90,110,150,215,275]
    },

    "_71.tif": {
        "R":[255,217,229,196,160,114,76,33,12],
        "G":[255,217,235,214,196,169,141,115,41],
        "B":[255,219,250,229,211,195,178,184,74],
        "v":[15,25,35,45,55,65,75,85,90]
    },

    "_207.tif": {
        "R":[53,47,41,35,63,147,233,232,229,224,191],
        "G":[151,231,247,244,241,238,235,144,48,0,0],
        "B":[253,250,180,90,30,24,19,14,9,147,61],
        "v":[0.0105,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.2]
    },

    "_207_3HH.tif": {
        "R":[53,47,41,35,63,147,233,232,229,224,191],
        "G":[151,231,247,244,241,238,235,144,48,0,0],
        "B":[253,250,180,90,30,24,19,14,9,147,61],
        "v":[0.0105,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.2]
    },

    "_228.tif": {
        "R":[217,176,102,79,168,217,255,255,255,255,255,153,153,171,190],
        "G":[248,250,247,250,245,242,201,138,105,89,120,102,51,26,0],
        "B":[248,247,237,184,0,0,10,23,31,89,212,255,204,133,61],
        "v":[5,15,25,35,45,55,65,75,85,95,105,115,125,135,140]
    },

    "_228_3HH.tif": {
        "R":[217,176,102,79,168,217,255,255,255,255,255,153,153,171,190],
        "G":[248,250,247,250,245,242,201,138,105,89,120,102,51,26,0],
        "B":[248,247,237,184,0,0,10,23,31,89,212,255,204,133,61],
        "v":[5,15,25,35,45,55,65,75,85,95,105,115,125,135,140]
    }

}

# ============================================================
# Directory preparation
# ============================================================

def prepare_directories():

    for directory in [PB_DIR, CAN_DIR]:

        if os.path.exists(directory):
            shutil.rmtree(directory)

        os.makedirs(directory)


    if os.path.exists(TIFF_DIR):
        shutil.rmtree(TIFF_DIR)

    os.makedirs(TIFF_DIR)


# ============================================================
# Download
# ============================================================

def download_file(url, output_file):

    print(f"Downloading:")
    print(f"  {url}")

    response = requests.get(
        url,
        timeout=180
    )

    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)

    print(
        f"  -> {output_file} "
        f"({len(response.content) / 1024 / 1024:.1f} MB)"
    )


# ============================================================
# Extract
# ============================================================

def extract_archive(archive, directory):

    print(f"Extracting {archive}")

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(directory)

    print("  -> extraction complete")


# ============================================================
# Find files recursively
# ============================================================

def get_files(directory):

    files = []

    for root, dirs, filenames in os.walk(directory):

        for filename in filenames:

            if filename.lower().endswith(".tif"):
                files.append(
                    os.path.join(root, filename)
                )

    return sorted(files)


# ============================================================
# Get filename from forecast date/hour
# ============================================================

def build_filename(idate, hour, d_code):

    d_time = (
        idate +
        datetime.timedelta(hours=hour)
    )

    return (
        "down_"
        + d_time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + "+00:00"
        + d_code
    )


# ============================================================
# Build lookup dictionary
# ============================================================

def build_file_lookup(directory):

    files = get_files(directory)

    lookup = {}

    for path in files:
        lookup[os.path.basename(path)] = path

    return lookup


# ============================================================
# Build colour scale
# ============================================================

def build_scale(scale):

    scale_rgb = np.column_stack(
        [
            scale["R"],
            scale["G"],
            scale["B"]
        ]
    ).astype(np.int16)

    scale_values = np.asarray(
        scale["v"],
        dtype=np.float32
    )

    return scale_rgb, scale_values


# ============================================================
# NumPy RGB classification
# ============================================================

def classify_raster(
    hxr,
    scale_rgb,
    scale_values
):

    # RGB bands
    rgb = np.asarray(
        hxr.isel(
            band=[0, 1, 2]
        ).values,
        dtype=np.int16
    )

    # (3, y, x) -> (y, x, 3)
    pixels = np.moveaxis(
        rgb,
        0,
        -1
    )

    # Calculate L1 distance from every pixel
    # to every colour in the AEMET scale.
    #
    # result shape:
    #
    #   (y, x, number_of_scale_values)
    #
    distances = np.abs(
        pixels[..., None, :] -
        scale_rgb
    ).sum(axis=-1)

    # Closest colour
    indices = np.argmin(
        distances,
        axis=-1
    )

    # Convert colour index to actual
    # meteorological value
    result = scale_values[indices]

    return result.astype(
        np.float32
    )


# ============================================================
# Mosaic PB + CAN
# ============================================================

def merge_forecast_files(
    pb_file,
    can_file
):

    print("  Opening PB:")
    print(f"    {pb_file}")

    pb = rxr.open_rasterio(
        pb_file,
        masked=True
    )

    print("  Opening CAN:")
    print(f"    {can_file}")

    can = rxr.open_rasterio(
        can_file,
        masked=True
    )

    # --------------------------------------------------------
    # Mosaic geographically
    # --------------------------------------------------------

    merged = merge_arrays(
        [pb, can]
    )

    # Close source datasets
    pb.close()
    can.close()

    return merged


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    prepare_directories()


    # --------------------------------------------------------
    # Download BOTH AEMET domains
    # --------------------------------------------------------

    download_file(
        PB_URL,
        PB_ARCHIVE
    )

    download_file(
        CAN_URL,
        CAN_ARCHIVE
    )


    # --------------------------------------------------------
    # Extract BOTH archives
    # --------------------------------------------------------

    extract_archive(
        PB_ARCHIVE,
        PB_DIR
    )

    extract_archive(
        CAN_ARCHIVE,
        CAN_DIR
    )


    # --------------------------------------------------------
    # Build file lookups
    # --------------------------------------------------------

    pb_files = build_file_lookup(
        PB_DIR
    )

    can_files = build_file_lookup(
        CAN_DIR
    )


    print()
    print(
        f"PB TIFF files:  {len(pb_files)}"
    )

    print(
        f"CAN TIFF files: {len(can_files)}"
    )


    # --------------------------------------------------------
    # Get forecast initialisation date
    # --------------------------------------------------------

    first_file = sorted(
        pb_files.keys()
    )[0]

    idate = datetime.datetime.strptime(
        first_file[5:24],
        "%Y-%m-%dT%H:%M:%S"
    )

    print()
    print(
        f"Forecast initialisation: {idate}"
    )


    # --------------------------------------------------------
    # Process products
    # --------------------------------------------------------

    for d_code in D_CODES:

        print()
        print("=" * 70)
        print(
            f"PROCESSING {d_code}"
        )
        print("=" * 70)


        scale_rgb, scale_values = build_scale(
            SCALES[d_code]
        )


        # ----------------------------------------------------
        # Forecast hours
        # ----------------------------------------------------

        for hour in range(
            N_FORECAST_HOURS
        ):

            filename = build_filename(
                idate,
                hour,
                d_code
            )


            # ------------------------------------------------
            # Locate PB and CAN files
            # ------------------------------------------------

            pb_file = pb_files.get(
                filename
            )

            can_file = can_files.get(
                filename
            )


            print()
            print(
                f"Hour {hour:02d}: "
                f"{filename}"
            )


            if pb_file is None:
                print(
                    "  PB:  MISSING"
                )
                continue

            if can_file is None:
                print(
                    "  CAN: MISSING"
                )
                continue


            print("  PB:  OK")
            print("  CAN: OK")


            # ------------------------------------------------
            # Merge PB + CAN
            # ------------------------------------------------

            merged = merge_forecast_files(
                pb_file,
                can_file
            )


            # ------------------------------------------------
            # Crop to combined area
            # ------------------------------------------------

            merged = merged.rio.clip_box(
                minx=LON_MIN,
                miny=LAT_MIN,
                maxx=LON_MAX,
                maxy=LAT_MAX
            )


            # ------------------------------------------------
            # RGB -> meteorological value
            # ------------------------------------------------

            result = classify_raster(
                merged,
                scale_rgb,
                scale_values
            )


            # ------------------------------------------------
            # Preserve band 4 metadata
            # ------------------------------------------------

            output_band = merged.isel(
                band=3
            ).copy()

            output_band.values = result


            # ------------------------------------------------
            # Write result
            # ------------------------------------------------

            output_file = os.path.join(
                TIFF_DIR,
                filename + "f"
            )

            output_band.rio.to_raster(
                output_file
            )


            print(
                f"  -> {output_file}"
            )


            # ------------------------------------------------
            # Free memory
            # ------------------------------------------------

            merged.close()


    print()
    print(
        "--------------------------------------------------"
    )

    print(
        "AEMET HARMONIE-AROME forecast "
        "successfully processed!"
    )

    print(
        "--------------------------------------------------"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()