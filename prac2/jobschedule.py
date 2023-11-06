
import random

JOBS = 5

def jobschedule(ids, dls, pro):
    
    # Aggregating the jobs
    jobs= [i for i in zip(ids, dls, pro)]
    
    # sorting the jobs in ascending order to the profit
    jobs.sort(key = lambda x: x[2], reverse= True)

    # finding out the max deadline
    max_deadline = max(dls)

    # Initializing the schedule tracker
    track = [None] * max_deadline

    total_prof = 0
    for id, dl, prof in jobs:
        for i in range(dl-1, -1, -1):
            if track[i] == None:
                track[i] = id
                total_prof += prof
                break
    sched = [i for i in track if i != None]

    return sched, total_prof


ids = [i for i in range(1, JOBS+1)]
dedline = [random.randint(1, JOBS+1) for _ in range(JOBS)]
profit = [random.randint(1, 50) for _ in range(JOBS)]
print(f"\nids: {ids}\ndeadlines: {dedline}\nProfit: {profit}\n")

schedule, profit = jobschedule(ids, dedline, profit)

print(f"Schedule: {schedule}\nTotal profit: {profit}\n")