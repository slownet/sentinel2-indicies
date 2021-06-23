"""
https://www.hatarilabs.com/ih-en/sentinel2-images-explotarion-and-processing-with-python-and-rasterio
"""
from typing import Optional

import numpy as np
import rasterio
from numpy import ndarray
from rasterio import DatasetReader

from plot import save_image

GENERATE_BW_IMG = True

RESOLUTION = 'R10m'  # R10m, R20m, R60m
IMAGES_FOLDER = f'images/{RESOLUTION}'

FILENAMES_MAPPING = {
    'R10m': {
        'B2': 'T33UXQ_20210619T100031_B02_10m.jp2',
        'B3': 'T33UXQ_20210619T100031_B03_10m.jp2',
        'B4': 'T33UXQ_20210619T100031_B04_10m.jp2',
        'B8': 'T33UXQ_20210619T100031_B08_10m.jp2',
    },
    'R20m': {
        'B2': 'T33UXQ_20210619T100031_B02_20m.jp2',
        'B3': 'T33UXQ_20210619T100031_B03_20m.jp2',
        'B4': 'T33UXQ_20210619T100031_B04_20m.jp2',
        'B8': 'T33UXQ_20210619T100031_B8A_20m.jp2',
        'B11': 'T33UXQ_20210619T100031_B11_20m.jp2',
    },
    'R60m': {
        'B2': 'T33UXQ_20210619T100031_B02_60m.jp2',
        'B3': 'T33UXQ_20210619T100031_B03_60m.jp2',
        'B4': 'T33UXQ_20210619T100031_B04_60m.jp2',
        'B8': 'T33UXQ_20210619T100031_B8A_60m.jp2',
        'B11': 'T33UXQ_20210619T100031_B11_20m.jp2',
    }
}


def compute_moisture(band8: ndarray) -> Optional[ndarray]:

    print('Computing moisture')

    if RESOLUTION == 'R10m':
        return None

    band_reader11: DatasetReader = rasterio.open(
        f'{IMAGES_FOLDER}/{FILENAMES_MAPPING[RESOLUTION]["B11"]}', driver='JP2OpenJPEG'
    )  # swir

    band11 = band_reader11.read(1).astype(float)

    c = band8 - band11
    d = band8 + band11

    moisture = np.divide(c, d, out=np.zeros_like(c), where=d != 0)

    return moisture


def compute_ndvi(band4: ndarray, band8: ndarray) -> ndarray:

    print('Computing NDVI')

    a = band8 - band4
    b = band8 + band4

    ndvi = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    return ndvi


def main():

    print(f'Script started ({GENERATE_BW_IMG=}, {RESOLUTION=}, {IMAGES_FOLDER=})')

    # Create image readers

    print('Decoding images')

    band_reader2: DatasetReader = rasterio.open(
        f'{IMAGES_FOLDER}/{FILENAMES_MAPPING[RESOLUTION]["B2"]}', driver='JP2OpenJPEG'
    )  # blue
    band_reader3: DatasetReader = rasterio.open(
        f'{IMAGES_FOLDER}/{FILENAMES_MAPPING[RESOLUTION]["B3"]}', driver='JP2OpenJPEG'
    )  # green
    band_reader4: DatasetReader = rasterio.open(
        f'{IMAGES_FOLDER}/{FILENAMES_MAPPING[RESOLUTION]["B4"]}', driver='JP2OpenJPEG'
    )  # red
    band_reader8: DatasetReader = rasterio.open(
        f'{IMAGES_FOLDER}/{FILENAMES_MAPPING[RESOLUTION]["B8"]}', driver='JP2OpenJPEG'
    )  # nir


    # Create images in form of numpy arrays with float elements

    print('Converting images into numpy arrays')

    band2 = band_reader2.read(1).astype(float)
    band3 = band_reader3.read(1).astype(float)
    band4 = band_reader4.read(1).astype(float)
    band8 = band_reader8.read(1).astype(float)

    # Compute NDVI and Moisture indices

    ndvi = compute_ndvi(band4, band8)
    moisture = compute_moisture(band8)

    # Plot and save images

    # plot_image(ndvi, cmap='terrain_r')
    # plot_image(moisture, cmap='gist_ncar_r')

    print('Saving NDVI index')
    save_image(ndvi, img_path='output/ndvi_index.png', cmap='terrain_r')

    if moisture is not None:
        print('Saving moisture index')
        save_image(moisture, img_path='output/moisture_index.png', cmap='gist_ncar_r')

    # See more color schemes here:
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    if GENERATE_BW_IMG:

        print('Generating B&W image')

        with rasterio.open(
                'output/sentinel2-whole-product-wb.tiff', 'w', driver='Gtiff',
                width=band_reader4.width, height=band_reader4.height, count=3,
                crs=band_reader4.crs, transform=band_reader4.transform, dtype=band_reader4.dtypes[0]
        ) as trueColor:
            trueColor.write(band_reader2.read(1), 3)  # blue
            trueColor.write(band_reader3.read(1), 2)  # green
            trueColor.write(band_reader4.read(1), 1)  # red


if __name__ == '__main__':
    main()
