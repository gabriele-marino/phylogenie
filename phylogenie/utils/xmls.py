import re
from xml.dom import minidom


def beautify_xml(xml: bytes) -> str:
    xml_str = minidom.parseString(xml).toprettyxml(indent="\t")
    xml_str = re.sub(r"\n\s*\n+", "\n", xml_str)
    xml_str = xml_str.strip()
    return xml_str
