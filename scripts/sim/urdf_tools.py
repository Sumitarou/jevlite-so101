"""Small URDF edits applied at launch (standard library only, so tests need no ROS).

The upstream SO-101 URDF stays unchanged; sim.launch.py passes the expanded robot description
through `recolor()` with configs/sim.yaml `appearance`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

ARM_MATERIAL = "jevlite_arm"
BASE_MATERIAL = "jevlite_base"


def _rgba(values) -> str:
    return " ".join(f"{float(v):g}" for v in values)


def recolor(urdf_xml: str, appearance: dict) -> str:
    """Give the 3D-printed parts the configured colours.

    Visuals using `printed_material` get BASE_MATERIAL on `base_link` (the fixed base) and
    ARM_MATERIAL everywhere else. Other materials (e.g. the black motors) are kept.
    Raises ValueError if the printed material is not used at all (wrong name in the config).
    """
    root = ET.fromstring(urdf_xml)
    printed = appearance["printed_material"]
    for name, rgba in ((ARM_MATERIAL, appearance["arm_rgba"]),
                       (BASE_MATERIAL, appearance["base_rgba"])):  # fmt: skip
        mat = ET.Element("material", {"name": name})
        ET.SubElement(mat, "color", {"rgba": _rgba(rgba)})
        root.insert(0, mat)
    changed = 0
    for link in root.findall("link"):
        target = BASE_MATERIAL if link.get("name") == appearance["base_link"] else ARM_MATERIAL
        for vis in link.findall("visual"):
            mat = vis.find("material")
            if mat is not None and mat.get("name") == printed:
                for child in list(mat):  # drop an inline colour, if any
                    mat.remove(child)
                mat.set("name", target)
                changed += 1
    if changed == 0:
        raise ValueError(f"no visual uses the material {printed!r}")
    return ET.tostring(root, encoding="unicode")
