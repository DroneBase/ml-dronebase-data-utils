from ml_dronebase_data_utils.visualize import draw_boxes
from ml_dronebase_data_utils.s3 import download_file, upload_file
from PIL import Image
from xml.dom import minidom
import os
import tempfile

if __name__ == "__main__":
    ortho_path = "s3://ml-solar-ortho-fault-detection/orthos/tiff/PA140004_Thermal.tif"
    annotation_path = "s3://ml-solar-ortho-fault-detection/orthos/annotations/PA140004_Thermal.xml"
    save_path = "s3://ml-solar-ortho-fault-detection/orthos/visual_validation/PA140004_Thermal_drawn.png"

    if "s3://" in ortho_path:
        # Download
        path = os.path.join(tempfile.gettempdir(),os.path.basename(ortho_path))
        download_file(ortho_path,path)
        ortho_path = path

    if "s3://" in annotation_path:
        # Download
        path = os.path.join(tempfile.gettempdir(),os.path.basename(annotation_path))
        download_file(annotation_path,path)
        annotation_path = path

    img = Image.open(ortho_path)

    parser = minidom.parse(annotation_path)

    annotations = parser.getElementsByTagName('object')

    boxes = []
    classes = []

    for a in annotations:
        xmin = int(a.getElementsByTagName('xmin')[0].firstChild.data)
        xmax = int(a.getElementsByTagName('xmax')[0].firstChild.data)
        ymin = int(a.getElementsByTagName('ymin')[0].firstChild.data)
        ymax = int(a.getElementsByTagName('ymax')[0].firstChild.data)
        boxes.append([xmin,ymin,xmax,ymax])
        class_name = a.getElementsByTagName('name')[0].firstChild.data
        classes.append(class_name)

    img_drawn = draw_boxes(img,boxes,classes)

    upload = False
    if "s3://" in save_path:
        orig_save_path = save_path
        save_path = os.path.join(tempfile.gettempdir(),os.path.basename(save_path))
        upload = True

    img_drawn.save(save_path)

    if upload:
        upload_file(save_path,orig_save_path,exist_ok=False)




    
