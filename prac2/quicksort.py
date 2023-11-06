
import random

ARRLEN = 25
UPLIM = 500


def quicksort(x):
    if len(x) <= 1:
        return x
    
    piv = random.choice(x)

    left = [i for i in x if i < piv]
    right = [i for i in x if i > piv]
    mid = [i for i in x if i == piv]

    return quicksort(left) + mid + quicksort(right)


arr = [random.randint(10, UPLIM) for _ in range(ARRLEN)]

out = quicksort(arr)

print(f"original arr: {arr}\nSorted arr: {out}")
