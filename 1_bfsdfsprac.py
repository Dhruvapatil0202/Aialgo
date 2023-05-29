graph = {
    1 : [2,3],
    2 : [4],
    3 : [2,5],
    4 : [],
    5 : [2,6],
    6 : [3]
}


def DFS(graph, node):

    def traverser(graph, visited, node):

        visited.append(node)

        for i in graph[node]:
            if i not in visited:
                traverser(graph, visited, i)

        pass
    
    visited = []
    
    traverser(graph, visited, node)

    print(visited)



def BFS(graph, node):

    def traverser(graph, visited, node, ex):

        visited.append(node)
        ex.append(node)

        while ex:
            c = ex.pop(0)
            seq.append(c)

            for i in graph[c]:

                if i not in visited:
                    visited.append(i)
                    ex.append(i)


    visited = []
    ex = []
    seq = []

    traverser(graph, visited, node, ex)

    print(seq)


# BFS(graph, 5)

DFS(graph, 1)