
def rec(x):
    if x == 1: return 1
    return x * rec(x-1)

def iter(x):
    out =1 
    for i in range(2, x+1):
        out *= i
    return out

n = 5
print(iter(n+1))