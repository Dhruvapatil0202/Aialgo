
def gcd(a, b):
    
    while b != 0:
        temp = a
        a = b
        b = temp % b
    
    return a

a, b = 56625, 235625
out = gcd(a, b)
print(f'The GCD is {out}')