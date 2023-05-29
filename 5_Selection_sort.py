
import random

arr = [random.randint(1,100) for _ in range(random.randint(5,10))]

print(arr)

for i in range(len(arr)):

    maxver = i

    for j in range(i, len(arr)):

        if arr[j] < arr[maxver]:
            maxver = j

    minver = i

    arr[minver] , arr[maxver] = arr[maxver], arr[minver]


print(arr)