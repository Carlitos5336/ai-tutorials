import math

class Edge:
    def __init__(self, n1, n2, max_velocidad: float, dirigido: bool) -> None:
        self.n1 = n1  # Nodo 1
        self.n2 = n2  # Nodo 2
        self.max_velocidad = max_velocidad
        self._dirigido = dirigido
        self._costo = self._calcular_costo()

    def _calcular_costo(self) -> float:
        """Calcula el costo (tiempo) entre los dos nodos."""
        dx = float(self.n1.lon) - float(self.n2.lon)
        dy = float(self.n1.lat) - float(self.n2.lat)
        distancia = math.hypot(dx, dy) * 100000  # Ajuste de escala
        velocidad_mps = self.max_velocidad / 3.6  # km/h a m/s
        return distancia / velocidad_mps if velocidad_mps > 0 else float('inf')

    def get_costo(self) -> float:
        """Devuelve el costo ya calculado de la arista."""
        return self._costo

    def dirigido(self) -> bool:
        """Indica si la arista es dirigida."""
        return self._dirigido

    def get_node(self, x: int):
        """Devuelve el nodo 1 o 2 según el valor de x (1 o 2)."""
        if x == 1:
            return self.n1
        elif x == 2:
            return self.n2
        else:
            raise ValueError("x debe ser 1 o 2")
