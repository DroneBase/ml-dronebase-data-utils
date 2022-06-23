from ml_dronebase_data_utils.convert_geojson import geo_to_voc

if __name__ == "__main__":
    ortho_path = "s3://ml-solar-ortho-fault-detection/orthos/tiff/PA140004_Thermal.tif"
    geojson_path = "s3://ml-solar-ortho-fault-detection/orthos/geojson/PA140004_Thermal.geojson"
    save_path = "s3://ml-solar-ortho-fault-detection/orthos/annotations/PA140004_Thermal.xml"

    geo_to_voc(ortho_path,geojson_path,save_path)
