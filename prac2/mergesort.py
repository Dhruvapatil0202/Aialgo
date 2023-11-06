
import random

ARRLEN = 10
UPLIM = 500

def mergesort(x):
    
    if len(x) > 1:

        m = len(x) // 2
        lh = x[:m]
        rh = x[m:]

        mergesort(lh)
        mergesort(rh)

        i = j = k = 0

        while i < len(lh) and j < len(rh):
            if lh[i] < rh[j]:
                x[k] = lh[i]
                i += 1
            else:
                x[k] = rh[j]
                j += 1
            k += 1
            

        while i < len(lh):
            x[k] = lh[i]
            k += 1
            i += 1
        
        while j < len(rh):
            x[k] = rh[j]
            k += 1
            j += 1


        
arr = [random.randint(10, UPLIM) for _ in range(ARRLEN)]

print(f"\nunsorted arr: {arr}")

mergesort(arr)

print(f"Sorted arr: {arr}\n")