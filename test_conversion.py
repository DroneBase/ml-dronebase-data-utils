from ml_dronebase_data_utils.convert_geojson import geo_to_voc

if __name__ == "__main__":
    ortho_path = "s3://ml-solar-ortho-fault-detection/orthos/tiff/PA140004_Thermal.tif"
    geojson_path = "s3://ml-solar-ortho-fault-detection/orthos/geojson/PA140004_Thermal.geojson"
    save_path = "s3://ml-solar-ortho-fault-detection/orthos/annotations/PA140004_Thermal.xml"
    class_attribute = "id"
    class_mapping = {
        0: 'Normal',
        1: 'Diode',
        2: 'Panel Offline',
        3: 'String Offline',
        4: 'Panel Off-Tilt',
        5: 'PID',
        6: 'Soiling',
        7: 'Shading',
        8: 'Reverse Polarity',
        9: 'Glass Broken',
        10: 'No Panel Data',
        11: 'Hot Cell',
        12: 'Hot Cell',
        13: 'Hot Cell',
        14: 'Multiple Hot Cells'
    }
    skip_classes = [0,1,2,4,5,6,7,8,9,10,14]
    geo_to_voc(ortho_path,geojson_path,save_path,class_attribute,class_mapping,skip_classes=skip_classes)
