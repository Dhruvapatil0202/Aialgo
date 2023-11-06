import random

CIT = 6

def bktrak(gp, path, vis, cur, cost):
    global mincost
    global finalpath

    if len(path) == CIT:

        # found a new minimum path
        if cost + gp[cur][path[0]] < mincost:
            mincost = cost + gp[cur][path[0]]
            finalpath = path[:]

        return
    
    for i in range(CIT):

        if not vis[i] and gp[cur][i] != 0:

            vis[i] = True
            path.append(i)

            bktrak(gp, path, vis, i, cost + gp[cur][i])

            vis[i] = False
            path.pop()

def tspinit(gp, s):
    
    vis = [False] * CIT
    vis[s] = True
    path = [s]
    bktrak(gp, path, vis, s, 0)
    return mincost

gp = [[0] * CIT for _ in range(CIT)]

for i in range(CIT):
    for j in range(i, CIT):
        if i == j:
            continue
        tem = random.randint(1, 99)
        gp[i][j] = gp[j][i] = tem

print(f'\nThe graph:\n')
[print(i) for i in gp]

start = 0   # should be within range (0, CIT - 1)
mincost = float('inf')
finalpath = []

mincost = tspinit(gp, start)

print(f'\nMincost: {mincost}\nFinal Path: {finalpath}\n')
