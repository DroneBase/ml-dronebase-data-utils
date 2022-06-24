from ml_dronebase_data_utils.convert_geojson import geo_to_voc

def run_geojson_conversion(**kwargs):
    '''
    Convert geojson format to pascal voc format

    Keyword Arguments

    ortho_path -> The ortho path, can be local/s3
    geojson -> The geojson path
    save_path -> The save path
    class_attribute -> The class attribute to use from the geojson for class labels
    class_mapping -> A plain txt file containing class mappings
    skip_classes -> Classes to skip, specify multiple
    rotated -> Use rotated bounding box, defaults to false

    '''

    ortho_path = kwargs.get('ortho_path',None)
    geojson = kwargs.get('geojson',None)
    save_path = kwargs.get('save_path',None)

    if ortho_path is None or geojson is None or save_path is None:
        print("You must specify ortho_path, anno_path and save_path")
        return 1
    
    class_attribute = kwargs.get('class_attribute',None)
    class_mapping = kwargs.get('class_mapping',None)
    default_class = kwargs.get('default_class','panel')
    skip_classes = kwargs.get('skip_classes',[])
    rotated = kwargs.get('rotated',False)

    #Call the function
    geo_to_voc(
        ortho_path,
        geojson,
        save_path,
        class_attribute,
        class_mapping,
        default_class,
        skip_classes,
        rotated
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert geojson to voc format data")

    parser.add_argument('--ortho-path',required=True,help="The ortho path, can be local/s3")
    parser.add_argument('--geojson',required=True,help="The geojson path")
    parser.add_argument('--save-path',required=True,help="The save path")
    parser.add_argument('--class-attribute',type=str,help="The class attribute to use from the geojson for class labels")
    parser.add_argument('--class-mapping',help="A plain txt file containing class mappings")
    parser.add_argument('--skip-classes',type=int,nargs='+',help="Classes to skip, specify multiple")
    parser.add_argument('--rotated',action='store_true',default=False,help="Use rotated bounding box, defaults to false")

    args = vars(parser.parse_args())

    # Read class mapping if provided
    class_mapping = args.get('class_mapping',None)
    if class_mapping is not None:
        mapping = {}
        with open('class_mapping') as cm:
            for line in cm:
                key,value = line.split(':',maxsplit=2)
                key = key.strip()
                value = value.strip('\n').strip()
                mapping[key] = value
        args['class_mapping'] = mapping

    run_geojson_conversion(args)
