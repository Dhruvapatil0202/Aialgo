import heapq

def greedy_shortest_path(graph, source):
    # Initialize distances to all vertices as infinity except for the source vertex
    distances = {vertex: float('inf') for vertex in graph}
    distances[source] = 0

    # Initialize a priority queue to store vertices and their corresponding distances
    pq = [(0, source)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        # Skip if the current distance is greater than the known shortest distance
        if current_distance > distances[current_vertex]:
            continue

        # Explore the neighbors of the current vertex
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            # If a shorter path is found, update the distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Example graph representation
graph = {
    'A': [('B', 5), ('C', 2)],
    'B': [('A', 5), ('C', 1), ('D', 3)],
    'C': [('A', 2), ('B', 1), ('D', 6)],
    'D': [('B', 3), ('C', 6)]
}

# Specify the source vertex
source_vertex = 'A'

# Call the greedy_shortest_path function to find the shortest paths from the source vertex
shortest_paths = greedy_shortest_path(graph, source_vertex)

# Print the shortest paths from the source vertex to all other vertices
for vertex, distance in shortest_paths.items():
    print(f"Shortest path from {source_vertex} to {vertex}: {distance}")
