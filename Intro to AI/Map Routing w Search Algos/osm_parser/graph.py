from .edge import Edge

class Graph:
    def __init__(self, data_ways) -> None:
        self.aristas = []
        self.adyacencia = {}

        for way in data_ways:
            if way.open:
                for i in range(len(way.nodes) - 1):
                    try:
                        max_velocidad = way.tags.max_speed
                        dirigido = way.tags.oneway
                        arista = Edge(way.nodes[i], way.nodes[i + 1], max_velocidad, dirigido)
                        self.agregar(arista)
                    except Exception as e:
                        # Could log the error: print(f"Error creating edge: {e}")
                        pass

    def agregar(self, arista: Edge) -> None:
        """Agrega una arista al grafo, si no existe aún."""
        if arista not in self.aristas:
            self.aristas.append(arista)
            self._agregar_adyacencia(arista)

    def get_ady(self) -> dict:
        """Devuelve el diccionario de adyacencia."""
        return self.adyacencia

    def _agregar_adyacencia(self, arista: Edge) -> None:
        """Añade los nodos adyacentes al diccionario de adyacencia."""
        n1 = arista.get_node(1).get_coord()
        n2 = arista.get_node(2).get_coord()
        costo = arista.get_costo()
        dirigido = arista.dirigido()

        # Añadir n1 → n2
        if n1 not in self.adyacencia:
            self.adyacencia[n1] = []
        self.adyacencia[n1].append([n2, costo])

        # Si no es dirigido, también añadir n2 → n1
        if not dirigido:
            if n2 not in self.adyacencia:
                self.adyacencia[n2] = []
            self.adyacencia[n2].append([n1, costo])
