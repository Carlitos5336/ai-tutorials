import folium

def draw_location(coords:list):
    # Función que traza la ruta encontrada por el algoritmo en el mapa

    m = folium.Map(location = [coords[0][0], coords[0][1]], tiles='OpenStreetMap' ,zoom_start=16)  # Crear el mapa

    folium.Marker(location=[coords[0][0], coords[0][1]], popup="Origen").add_to(m)  # Marcador de origen

    folium.Marker(
            location=[coords[len(coords)-1][0], coords[len(coords)-1][1]],  
            icon=folium.Icon(color='red'),
             popup="Destino").add_to(m) # Marcador de destino
   
    folium.PolyLine(coords).add_to(m) # Se traza la ruta a tomar

    m.save('index.html')

def empty_map():

    m = folium.Map(location = [18.4679962,-69.2974377], tiles='OpenStreetMap' ,zoom_start=14.5)
    m.save('index.html')