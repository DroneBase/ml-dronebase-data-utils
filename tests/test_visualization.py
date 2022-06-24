from ml_dronebase_data_utils.visualize_converted_geojson import visualize

def test_visualization():
    visualize({
        'ortho_path':  "s3://ml-solar-ortho-fault-detection/orthos/tiff/PA140004_Thermal.tif",
        'annotation_path': "s3://ml-solar-ortho-fault-detection/orthos/annotations/PA140004_Thermal.xml",
        'save_path': "s3://ml-solar-ortho-fault-detection/orthos/visual_validation/PA140004_Thermal_drawn.png"
    })
