from dataclasses import dataclass

@dataclass
class Node:

    id: str
    lat: str
    lon: str

    def get_coord(self):
        return  (float(self.lat), float(self.lon))
    def __hash__(self):
        return hash(Node)
