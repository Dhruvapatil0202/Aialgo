

def fibonacci(n):
    if n == 1: return 1, [0]
    elif n == 2: return 2, [0, 1]

    steps = 2 
    ser = [0, 1]

    for _ in range(n-2):
        ser.append(ser[-1] + ser[-2])
        steps += 1

    return steps, ser

n = 10
steps, ser = fibonacci(n)
print(f"\nNo. of steps: {steps}\nFib series: {ser}\n")