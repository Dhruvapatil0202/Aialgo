
import heapq

graph = {
    'a': [('b',5), ('c', 2)],
    'b': [('a',5), ('c',1), ('d', 3)],
    'c': [('a', 2), ('b',1), ('d', 6)],
    'd': [('b', 3), ('c', 6)]
}



def dijkstra(graph, start):

    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    pq = [(0, start)]

    while pq:

        current_distance, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances



for i in graph.keys():

    print()
    distances = dijkstra(graph, i)
    for vertex, distance in distances.items():
        print(f"Shortest distance from {i} to {vertex}: {distance}")

    print()


