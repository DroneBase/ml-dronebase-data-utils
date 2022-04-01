from tempfile import NamedTemporaryFile

from ml_dronebase_data_utils import PascalVOCWriter


def test_writer():
    with NamedTemporaryFile() as f:
        writer = PascalVOCWriter(path=f.name, width=128, height=128)
        writer.addObject(name="test", xmin=0, ymin=0, xmax=10, ymax=10)
        writer.save(annotation_path=f.name)
