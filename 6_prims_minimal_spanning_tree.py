import heapq

graph = {
    'a': [('b', 2), ('c', 3)],
    'b': [('a', 2), ('c', 1), ('d', 4)],
    'c': [('a', 3), ('b', 1), ('d', 2)],
    'd': [('b', 4), ('c', 2)]
}

start_index = 'a'

def prim_mst(graph, start_index):
    mst = []
    visited = set([start_index])
    pq = []

    for neighbour, weight in graph[start_index]:
        heapq.heappush(pq, (weight, start_index, neighbour))

    while pq:
        weight, u, v = heapq.heappop(pq)

        if v not in visited:
            visited.add(v)
            mst.append((u,v))

            for neighbor, weight in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(pq, (weight, v, neighbor))

    return mst


mst = prim_mst(graph, start_index)

for u, v in mst:
    print(f"{u} -- {v}")


    