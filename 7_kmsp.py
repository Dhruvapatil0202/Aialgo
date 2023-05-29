
class UnionFind:

    def __init__(self, n):

        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False
    
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y

        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x

        else: 
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True
    
def kruskal_mst(graph):

    edges = []
    for u in graph:
        for  v, weight in graph[u]:
            edges.append((weight, u, v))

    edges.sort()

    num_vertices = len(graph)
    mst = []
    uf = UnionFind(num_vertices)

    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u,v))
        
    return mst


graph = {
    0: [(1,2), (2,3)],
    1: [(0,2), (2,1), (3,4)],
    2: [(0,3), (1,1), (3,2)],
    3: [(1,4), (2,2)]

}

mst = kruskal_mst(graph)

print("Minimal spanning Turee:")

for u,v in mst:
    print(f"{u} -- {v}")

