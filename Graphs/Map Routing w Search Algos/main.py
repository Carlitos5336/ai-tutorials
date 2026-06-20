from osm_parser.xml_data_processor import (
    find_nodes,
    find_ways,
    data_ways
)
from osm_parser.graph import Graph
from front import UI

def main() -> None:

    print("Starting node search operation")
    find_nodes()
    print("Finishing node search operation successfully")
    print("Starting way search operation")
    find_ways()
    print("Finishing way search operation successfully")
    g = Graph(data_ways)

    ady = g.get_ady()

    app = UI.QApplication([])
    UI.Map_Displayer(ady)
    app.exec_()

main()