

import random

NO = 5

def knapsack(ind, curwet, curval, itms):
    global max_profit, sel_items

    if ind == NO or curwet > cap:
        return
    
    if curval > max_profit:
        max_profit = curval
        sel_items = itms

    knapsack(ind + 1, curwet + wets[ind], curval + vals[ind], itms + [ind+1])

    knapsack(ind + 1, curwet, curval, itms)

wets = [random.randint(1, 10) for _ in range(NO)]
vals = [random.randint(10, 100) for _ in range(NO)]
cap = int(sum(wets) * random.random() + 0.01)
max_profit, sel_items = 0, []

print(f"\nwets: {wets} \nvals: {vals} \ncapicity: {cap}\n")

knapsack(0, 0, 0, [])

print(f"Max profit: {max_profit}\nSelected Items: {sel_items}\n")
