"""
Modified from https://github.com/AndrewCarterUK/pascal-voc-writer
"""
import os

from jinja2 import Environment, PackageLoader


class PascalVOCWriter:
    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        depth: int = 3,
        database: str = "Unknown",
        segmented: int = 0,
    ) -> None:
        environment = Environment(
            loader=PackageLoader("pascal_voc_writer", "templates"),
            keep_trailing_newline=True,
        )
        self.annotation_template = environment.get_template("annotation.xml")
        abspath = os.path.abspath(path)
        self.template_parameters = {
            "path": abspath,
            "filename": os.path.basename(abspath),
            "folder": os.path.basename(os.path.dirname(abspath)),
            "width": width,
            "height": height,
            "depth": depth,
            "database": database,
            "segmented": segmented,
            "objects": [],
        }

    def addObject(
        self,
        name: str,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        pose: str = "Unspecified",
        truncated: int = 0,
        difficult: int = 0,
    ) -> None:
        self.template_parameters["objects"].append(  # type: ignore[attr-defined]
            {
                "name": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "pose": pose,
                "truncated": truncated,
                "difficult": difficult,
            }
        )

    def save(self, annotation_path: str) -> None:
        with open(annotation_path, "w") as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)
