import ee
import ssl
import time
import pandas as pd
from pathlib import Path
from pathlib import Path
import numpy as np

# Earth Engine Initialization
ee.Initialize()

# Utility functions
def get_tif_files(image_path):
    """
    Get all the .tif files in the image folder.
    """
    return list(image_path.glob('*.tif'))

def load_clean_yield_data(yield_data_filepath):
    """
    Cleans the yield data by making sure any NaN values in the columns we care about
    are removed.
    """
    important_columns = ["Year", "State ANSI", "County ANSI", "Value"]
    yield_data = pd.read_csv(yield_data_filepath).dropna(
        subset=important_columns, how="any"
    )
    return yield_data
def _append_precipitation_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Rename the band
    previous = ee.Image(previous)
    current = current.select(['prcp'])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )
def _append_temp_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Rename the band
    previous = ee.Image(previous)
    current = current.select(['LST_Day_1km','LST_Night_1km'])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )
def _append_vegetation_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Rename the band
    previous = ee.Image(previous)
    current = current.select(['NDVI','EVI'])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )
def _append_mask_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Rename the band
    previous = ee.Image(previous)
    current = current.select(['LC_Type1'])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )
def _append_im_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Author: Jamie Vleeshouwer

    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0, 1, 2, 3, 4, 5, 6])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )
# MODISExporter class
class MODISExporter:
    def __init__(self, locations_filepath = None, collection_id = None):
        self.locations = load_clean_yield_data(locations_filepath)
        self.collection_id = collection_id

    @staticmethod
    def _export_one_image(img, folder, name, region, scale, crs):
        print(f"Exporting to {folder}/{name}")
        task_dict = {
            "driveFolder": folder,
            "driveFileNamePrefix": name,
            "scale": scale,
            "crs": crs,
        }
        if region is not None:
            task_dict.update({"region": region})
        task = ee.batch.Export.image(img, name, task_dict)
        task.start()
        while task.status()["state"] == "RUNNING":
            print("Running...")
            # Perhaps task.cancel() at some point.
            time.sleep(10)

        print(f"Done: {task.status()}")

    def export(self, folder_name, data_type, coordinate_system="EPSG:4326", scale=1000,check_if_done=False, download_folder=None,min_img_val=None,max_img_val=None,):
        if check_if_done:
            if download_folder is None:
                download_folder = Path("data") / folder_name
                already_downloaded = get_tif_files(download_folder)
        imgcoll = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(ee.Geometry.Rectangle(-91.5, 42.5, -87.5, 36.8))
            .filterDate("2017-12-31", "2022-12-31")
        )
        datatype_to_func = {
            "precipitation": _append_precipitation_band,
            "image":_append_im_band,
            'temp':_append_temp_band,
            'mask':_append_mask_band,
            "vegetation":_append_vegetation_band
        }

        img = imgcoll.iterate(datatype_to_func[data_type])
        img = ee.Image(img)
        if min_img_val is not None:
            # passing en ee.Number creates a constant image
            img_min = ee.Image(ee.Number(min_img_val))
            img = img.min(img_min)
        if max_img_val is not None:
            img_max = ee.Image(ee.Number(max_img_val))
            img = img.max(img_max)

        # Define the region using the counties FeatureCollection
        region = ee.FeatureCollection("TIGER/2018/Counties")
        region = region.map(lambda feature: feature.set("COUNTYFP", ee.Number.parse(feature.get("COUNTYFP"))))
        region = region.map(lambda feature: feature.set("STATEFP", ee.Number.parse(feature.get("STATEFP"))))

        count = 0

        for state_id, county_id in np.unique(self.locations[["State ANSI", "County ANSI"]].values, axis=0):
            file_region = region.filterMetadata("COUNTYFP", "equals", int(county_id)) \
                                .filterMetadata("STATEFP", "equals", int(state_id))
            file_region = ee.Feature(file_region.first())
            processed_img = img.clip(file_region)
            file_region = None

            # Define the export name based on state and county
            export_name = f"{int(state_id)}_{int(county_id)}"

            # Check if this image has already been downloaded
            if check_if_done and f"{export_name}.tif" in already_downloaded:
                print(f"{export_name}.tif already downloaded! Skipping")
                continue
            while True:
                try:
                    self._export_one_image(
                        processed_img,
                        folder_name,
                        export_name,
                        file_region,
                        scale,
                        coordinate_system,
                    )
                except (ee.ee_exception.EEException, ssl.SSLEOFError):
                    print(f"Retrying State {int(state_id)}, County {int(county_id)}")
                    time.sleep(10)
                    continue
                break

            count += 1
        print(f"Finished Exporting {count} files!")

# Example usage
def main():
    download_folder = [None] * 5
    locations_filepath = Path("data/corn_yield_data_2018_2022.csv")
    modis_collection_id = "MODIS/061/MOD13A1"
    exporter_modis = MODISExporter(locations_filepath, modis_collection_id)
    exporter_modis.export(
        folder_name="crop_yield-data_vegetation",
        data_type="vegetation",
        coordinate_system="EPSG:4326",
        scale=1000,
        check_if_done=True,
        download_folder=download_folder[0]
    )
    collection_id = "NASA/ORNL/DAYMET_V4"
    exporter = MODISExporter(locations_filepath, collection_id)
    
    # Example call to export function
    exporter.export(
        folder_name="crop_yield-data_precipitation",
        data_type="precipitation",
        coordinate_system="EPSG:4326",
        scale=1000,
        check_if_done=True,
        download_folder = download_folder[1],
        min_img_val=0,  # Set the minimum value for the image export
        max_img_val=544 # Set the maximum value for the image export
    )
    modis_collection_id = "MODIS/061/MOD09A1"
    exporter_modis = MODISExporter(locations_filepath, modis_collection_id)
    exporter_modis.export(
        folder_name="crop_yield-data_image",
        data_type="image",
        coordinate_system="EPSG:4326",
        scale=1000,
        check_if_done=True,
        download_folder=download_folder[2],
        min_img_val=100,  # Set the minimum value for the image export
        max_img_val=16000 # Set the maximum value for the image export
    )
    modis_collection_id = "MODIS/061/MOD11A2"
    exporter_modis = MODISExporter(locations_filepath, modis_collection_id)
    exporter_modis.export(
        folder_name="crop_yield-data_temp",
        data_type="temp",
        coordinate_system="EPSG:4326",
        scale=1000,
        check_if_done=True,
        download_folder=download_folder[3],
        min_img_val=750,  # Set the minimum value for the image export
        max_img_val=65535  # Set the maximum value for the image export
    )
    modis_collection_id = "MODIS/061/MCD12Q1"
    exporter_modis = MODISExporter(locations_filepath, modis_collection_id)
    exporter_modis.export(
        folder_name="crop_yield-data_mask",
        data_type="mask",
        coordinate_system="EPSG:4326",
        scale=1000,
        check_if_done=True,
        download_folder=download_folder[4]
    )
    print("Done exporting! Download the folders from your Google Drive")

if __name__ == "__main__":
    main()