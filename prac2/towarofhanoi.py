

def toh(ind, curr, aux, target):
    
    if ind == 1:
        print(f'The disc 1 moved from {curr} to {target}')
        return 
    
    toh(ind-1, curr, target, aux)

    print(f'The disc {ind} moved from {curr} to {target}')

    toh(ind-1, aux, curr, target)
    

n = 3
toh(n, 'A', 'B', 'C')
