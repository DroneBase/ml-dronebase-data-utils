from ml_dronebase_data_utils.visualize import draw_boxes
from ml_dronebase_data_utils.s3 import download_file, upload_file
from PIL import Image
from xml.dom import minidom
import os
import tempfile
import argparse

def visualize(**kwargs):
    '''
    Visualize converted xml annotations
    keyword Arguments
    
    ortho_path -> The path to the ortho
    anno_path -> The path to the annotation
    save_path -> The save path
    draw_labels -> Draw the labels or not, defaults to False

    '''
    ortho_path = kwargs.get('ortho_path',None)
    anno_path = kwargs.get('anno_path',None)
    save_path = kwargs.get('save_path',None)
    draw_labels = kwargs.get('draw_labels',False)

    if ortho_path is None or anno_path is None or save_path is None:
        print("You must specify ortho_path, anno_path and save_path")
        return 1
    
    # Create a temporary directory which is cleaned up after use
    with tempfile.TemporaryDirectory() as tmpdir:
        if "s3://" in ortho_path:
            # Download
            path = os.path.join(tmpdir,os.path.basename(ortho_path))
            download_file(ortho_path,path)
            ortho_path = path
        
        if "s3://" in annotation_path:
            # Download
            path = os.path.join(tmpdir,os.path.basename(annotation_path))
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
            if draw_labels:
                class_name = a.getElementsByTagName('name')[0].firstChild.data
                classes.append(class_name)

        img_drawn = draw_boxes(img,boxes,classes)

        upload = False
        if "s3://" in save_path:
            orig_save_path = save_path
            save_path = os.path.join(tmpdir,os.path.basename(save_path))
            upload = True

        img_drawn.save(save_path)

        if upload:
            upload_file(save_path,orig_save_path,exist_ok=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize converted geojson for quick visual inspection")

    parser.add_argument('--ortho-path','-o',required=True,help="The ortho path, can be local/s3")
    parser.add_argument('--anno-path','-a',required=True,help="The ortho path, can be local/s3")
    parser.add_argument('--save-path','-s',required=True,help="The ortho path, can be local/s3")
    parser.add_argument('--draw-labels','-d',action='store_true',default=False,help="Draw the class labels")

    args = vars(parser.parse_args())

    visualize(**args)
