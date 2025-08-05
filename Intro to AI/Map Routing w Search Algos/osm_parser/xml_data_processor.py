import xml.etree.ElementTree as ET
import xmltodict
from .node import Node
from .way import Tags, Way

archivo_xml = ET.parse('data/san_pedro.osm')
root = archivo_xml.getroot()

nodes = root.findall('node')
data_nodes = []
ways = root.findall('way')
data_ways = []

def find_nodes() -> None:
    for node in nodes:
        nodo = Node(
            node.attrib['id'],
            node.attrib['lat'], 
            node.attrib['lon']
        )
        data_nodes.append(nodo)

def find_ways() -> None:
    for way in ways:
        decoded_way = ET.tostring(way)
        parsed_data = xmltodict.parse(decoded_way)
        way_data = parsed_data["way"]
        
        via = Way(way_data["@id"])
        
        for nd in way_data.get("nd", []):
            ref_id = nd["@ref"]
            matching_nodes = [n for n in data_nodes if n.id == ref_id]
            via.nodes.extend(matching_nodes)

        nd_refs = way_data.get("nd", [])
        if len(nd_refs) >= 2 and nd_refs[0]["@ref"] == nd_refs[-1]["@ref"]:
            via.open = False

        tags = Tags()
        tags.highway = "residential"
        tags.max_speed = 30

        try:
            if via.open:
                for tag_data in way_data.get("tag", []):
                    try:
                        tag = tag_data["@k"]
                        value = tag_data["@v"]

                        if tag == "highway":
                            tags.highway = value
                            tags.max_speed = {
                                "residential": 30,
                                "primary": 100,
                                "secondary": 50,
                                "tertiary": 60
                            }.get(value, 30)

                        elif tag == "name":
                            tags.name = value

                        elif tag == "oneway":
                            tags.oneway = value == "yes"

                    except (TypeError, KeyError):
                        tags.highway = "residential"
                        tags.max_speed = 30

                if not tags.highway:
                    tags.highway = "residential"
                    tags.max_speed = 30

            via.tags = tags
            data_ways.append(via)

        except Exception as e:
            # Could log the error: print(f"Error processing way {via.id}: {e}")
            data_ways.append(via)
