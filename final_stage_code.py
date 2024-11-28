import numpy as np
import pandas as pd
import random

import numpy as np
import pandas as pd
from random import shuffle, randint, random
import random
# Provided data (assuming data has been corrected and is ready)
import pandas as pd

data = {
"CUST NO."  : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,],
"XCOORD."  : [35,41,35,55,55,15,25,20,10,55,30,20,50,30,15,30,10,5,20,15,45,45,45,55,65,65,45,35,41,64,40,31,35,53,65,63,2,20,5,60,40,42,24,23,11,6,2,8,13,6,47,49,27,37,57,63,53,32,36,21,17,12,24,27,15,62,49,67,56,37,37,57,47,44,46,49,49,53,61,57,56,55,15,14,11,16,4,28,26,26,31,15,22,18,26,25,22,25,19,20,18,],
"YCOORD."  : [35,49,17,45,20,30,30,50,43,60,60,65,35,25,10,5,20,30,40,60,65,20,10,5,35,20,30,40,37,42,60,52,69,52,55,65,60,20,5,12,25,7,12,3,14,38,48,56,52,68,47,58,43,31,29,23,12,12,26,24,34,24,58,69,77,77,73,5,39,47,56,68,16,17,13,11,42,43,52,48,37,54,47,37,31,22,18,18,52,35,67,19,22,24,27,24,27,21,21,26,18,],
"DEMAND"  : [0,10,7,13,19,26,3,5,9,16,16,12,19,23,20,8,19,2,12,17,9,11,18,29,3,6,17,16,16,9,21,27,23,11,14,8,5,8,16,31,9,5,5,7,18,16,1,27,36,30,13,10,9,14,18,2,6,7,18,28,3,13,19,10,9,20,25,25,36,6,5,15,25,9,8,18,13,14,3,23,6,26,16,11,7,41,35,26,9,15,3,1,2,22,27,20,11,12,10,9,17,],
"READY TIME"  : [0,161,50,116,149,34,99,81,95,97,124,67,63,159,32,61,75,157,87,76,126,62,97,68,153,172,132,37,39,63,71,50,141,37,117,143,41,134,83,44,85,97,31,132,69,32,117,51,165,108,124,88,52,95,140,136,130,101,200,18,162,76,58,34,73,51,127,83,142,50,182,77,35,78,149,69,73,179,96,92,182,94,55,44,101,91,94,93,74,176,95,160,18,188,100,39,135,133,58,83,185,],
"DUE DATE"  : [230,171,60,126,159,44,109,91,105,107,134,77,73,169,42,71,85,167,97,86,136,72,107,78,163,182,142,47,49,73,81,60,151,47,127,153,51,144,93,54,95,107,41,142,79,42,127,61,175,118,134,98,62,105,150,146,140,111,210,28,172,86,68,44,83,61,137,93,152,60,192,87,45,88,159,79,83,189,106,102,192,104,65,54,111,101,104,103,84,186,105,170,28,198,110,49,145,143,68,93,195,],
"SERVICE TIME"  : [0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,],
}
time_of_work=10
nbr_of_nurse=10
max_nbr_of_served_clients=15
time_spent=10
nbr_client=100
# Convert data to DataFrame
df = pd.DataFrame(data)


# Calculate Euclidean distance from the first candidate
first_candidate_coords = (40,50)
df['DISTANCE'] = df.apply(lambda row: np.sqrt((row['XCOORD.'] - first_candidate_coords[0])**2 + (row['YCOORD.'] - first_candidate_coords[1])**2), axis=1)
df['DISTANCE']*=-1;
NORM=(df['DISTANCE'].max() - df['DISTANCE'].min())
# Normalize the distance
df['DISTANCE_norm'] = (df['DISTANCE'] - df['DISTANCE'].min()) / NORM

# Normalize the demand
df['DEMAND_norm'] = (df['DEMAND'] - df['DEMAND'].min()) / (df['DEMAND'].max() - df['DEMAND'].min())
df['cost']=df['SERVICE TIME']+df['DISTANCE']
df['cost']=(df['cost']-df['cost'].min())/df['cost'].max()
# Assuming equal weights for simplicity
weights = np.array([1/3, 1/3,1/3])  # Equal weights for DISTANCE and DEMAND

# Weighted normalized values
df['DISTANCE_weighted'] = df['DISTANCE_norm'] * weights[0]
df['DEMAND_weighted'] = df['DEMAND_norm'] * weights[1]
df['cost']*=weights[2]
# Ideal vector (max values for each criterion)
ideal_vector = np.array([df['DISTANCE_weighted'].max(), df['DEMAND_weighted'].max()],df['cost'].max())

# Step 3: Perform Pairwise Comparisons (AHP)

# Example pairwise comparison matrix (you should adjust this based on your judgment)
comparison_matrix = np.array([
    [1, 1/3 , 2 ],    # DISTANCE is 3 times more important than DEMAND
    [1/3, 1 ,2 ], # DEMAND is 1/3 as important as DISTANCE
    [1/2 , 1/2 , 1 ]
])

# Normalize the pairwise comparison matrix
weights_criteria = np.mean(comparison_matrix, axis=1) / np.sum(np.mean(comparison_matrix, axis=1))

# Step 4: Calculate Scores

# Calculate scores based on AHP weights
df['score'] = weights_criteria[0] * df['DISTANCE_weighted'] + weights_criteria[1] * df['DEMAND_weighted'] + weights_criteria[2] * df['cost']

# Rank the candidates based on the scores
df['rank'] = df['score'].rank(ascending=False)

# Sort the candidates by rank
ranked_df = df.sort_values(by='rank')
ranked_df_reversed = ranked_df[::-1]
# Output the ranked candidates
print(ranked_df[['CUST NO.', 'score', 'rank']])
df=ranked_df







import numpy as np
import pandas as pd
import random
from math import sqrt

# Example DataFrame for nurses' coordinates (Replace with actual data)
df_nurse = pd.DataFrame({
    'nurse NO': list(range(1, nbr_of_nurse + 1)),
    'XCOORD.': [40 for _ in range(nbr_of_nurse)],  # Example coordinates
    'YCOORD.': [50 for _ in range(nbr_of_nurse)],
    'nurse\'s nbres of hours available': [480] * nbr_of_nurse  # Example availability in minutes
})

# Dummy AHP weights
ahp_weights = {
    'distance': 1/3,
    'workload_balance': 1/3,
    'time_window_adherence': 1/3
}
# Dummy client time windows
client_time_windows = {client: (df.loc[df['CUST NO.'] == client, 'READY TIME'].values[0], df.loc[df['CUST NO.'] == client, 'DUE DATE'].values[0]) for client in df['CUST NO.']}

def calculate_distance(nurse, client):
    # Get coordinates for nurse and client
    nurse_coords = df_nurse[df_nurse['nurse NO'] == nurse][['XCOORD.', 'YCOORD.']].values[0]
    client_coords = df[df['CUST NO.'] == client][['XCOORD.', 'YCOORD.']].values[0]

    # Calculate Euclidean distance
    distance = np.sqrt((nurse_coords[0] - client_coords[0])**2 + (nurse_coords[1] - client_coords[1])**2)
    return distance

def find_start_time(nurse, client, time_spent, schedule):
    # Extract the client's ready time and due date
    client_window_start = int(df.loc[df['CUST NO.'] == client, 'READY TIME'].values[0])
    client_window_end = int(df.loc[df['CUST NO.'] == client, 'DUE DATE'].values[0])

    # Search for a suitable start time within the client's time window
    for start_time in range(client_window_start, client_window_end - int(time_spent) + 1):
        tr = 0
        for x in range(start_time, start_time + time_of_work):
            if schedule[x] == 1:
                tr += 1
                break
        if tr == 0:
            return start_time
    return None

def mark_nurse_busy(nurse, start_time, time_spent, schedule):
    for i in range(start_time, start_time + int(time_spent)):
        schedule[i] = 1

def calculate_time(nurse, client):
    distance = calculate_distance(nurse, client)
    travel_time = distance / 2  # Assuming average speed is 2 units per minute
    service_time = df[df['CUST NO.'] == client]['SERVICE TIME'].values[0]
    return travel_time + service_time

def evaluate_time_window_adherence(solution, nurse_schedules):
    penalty = 0
    clients_served = set()
    for nurse, clients in solution.items():
        i = 0
        while i < len(clients):
            client = clients[i]
            if client in clients_served:
                clients.pop(i)
                continue
            if not is_within_time_window(nurse, client, nurse_schedules):
                clients.pop(i)
                continue
            start_time = find_start_time(nurse, client, time_of_work, nurse_schedules[nurse])
            if start_time is None:
                clients.pop(i)
                continue
            mark_nurse_busy(nurse, start_time, time_of_work, nurse_schedules[nurse])
            clients_served.add(client)
            i += 1
    return penalty

def is_within_time_window(nurse, client, nurse_schedules):
    # Get nurse's schedule
    nurse_schedule = nurse_schedules[nurse]
    # Get client's time window
    client_window_start, client_window_end = client_time_windows[client]

    # Check if nurse can start serving the client within the client's time window
    for start_time in range(client_window_start, client_window_end - df[df['CUST NO.'] == client]['SERVICE TIME'].values[0] + 1):
        tr = 0
        for x in range(start_time, start_time + time_of_work):
            if nurse_schedule[x] == 1:
                tr += 1
                break
        if tr == 0:
            return True
    return False

def calculate_the_distance(nurse_clients):
    x, y = 40, 50
    total_distance = 0
    for client in nurse_clients:
        ox = df['XCOORD.'][client - 1] - x
        oy = df['YCOORD.'][client - 1] - y
        total_distance += sqrt(ox**2 + oy**2)
        x, y = df['XCOORD.'][client - 1], df['YCOORD.'][client - 1]
    return total_distance

def fitness(solution):
    total_distance = 0
    max_work_time = 480
    workloads = [0]
    nbr_of_served_client = 0
    nurse_schedules = {i: [0] * 2000 for i in range(1, nbr_of_nurse + 1)}
    nbr_of_clients_per_nurse = 0
    time_window_adherence_score = evaluate_time_window_adherence(solution, nurse_schedules) * 20

    for nurse, clients in solution.items():
        while len(clients) > max_nbr_of_served_clients:
            clients.pop()
        nbr_of_served_client += len(clients)
        total_distance += calculate_the_distance(clients)
    # AHP-based evaluation
    distance_score = total_distance * ahp_weights['distance']
    workload_balance_score = (max(workloads) - min(workloads)) * ahp_weights['workload_balance']
    ahp_score = distance_score + workload_balance_score + time_window_adherence_score
    if nbr_of_served_client == 0:
        return 1e18
    mp = []
    for x,y in solution.items() :
      for i in y :
         mp.append(i)
    for i in range(1,100) :
      if i in mp :
           continue
      for x,y in solution.items() :
        if len(y)>=10 :
          continue
        h=find_start_time(x,i,time_spent,nurse_schedules[x])
        if h!=None :
          solution[x].append(i)
          mark_nurse_busy(x,h,time_spent,nurse_schedules[x])
          nbr_of_served_client+=1
    nbr_of_served_client *= 10
    composite_score = ahp_score / nbr_of_served_client

    return composite_score

def mutate(solution):
    nurses = list(solution.keys())
    nurse1, nurse2 = random.sample(nurses, 2)

    solution_int = {k: v[:] for k, v in solution.items()}  # Make a deep copy

    if solution[nurse1] and solution[nurse2]:
        client1 = random.choice(solution[nurse1])
        client2 = random.choice(solution[nurse2])

        solution_int[nurse1].remove(client1)
        solution_int[nurse2].remove(client2)

        if client1 not in solution_int[nurse2]:
            solution_int[nurse2].append(client1)
        if client2 not in solution_int[nurse1]:
            solution_int[nurse1].append(client2)

        if fitness(solution_int) < fitness(solution):
            return solution_int
    return solution

def crossover(parent1, parent2):
    child = {nurse: [] for nurse in parent1.keys()}
    for nurse in parent1.keys():
        if random.random() > 0.5:
            child[nurse] = parent1[nurse]
        else:
            child[nurse] = parent2[nurse]

    return child

def initialize_population(size):
    population = []

    for _ in range(size):
        solution = {nurse: [] for nurse in df_nurse['nurse NO']}
        clients = list(df['CUST NO.'])
        random.shuffle(clients)

        for client in clients:
            nurse = random.choice(list(solution.keys()))
            if client not in [item for sublist in solution.values() for item in sublist]:
                solution[nurse].append(client)

        population.append(solution)

    return population

def genetic_algorithm_run(population_size, generations, mutation_rate):
    population = initialize_population(population_size)

    for generation in range(generations):
        population = sorted(population, key=lambda x: fitness(x))
        next_generation = population[:population_size // 2]

        while len(next_generation) < population_size:
            if len(next_generation) < 2:
                break  # Ensure there are at least two parents to sample

            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                child = mutate(child)

            if fitness(child) < float('inf'):
                next_generation.append(child)

        population = next_generation

        best_solution = min(population, key=lambda x: fitness(x))
        print(f"Generation {generation}, Best Fitness: {fitness(best_solution)}")

    return best_solution

# Parameters
pop_size = 20
generations = 30
mutation_rate = 0.1

# Run GA
best_solution = genetic_algorithm_run(pop_size, generations, mutation_rate)
mpp = set()
for nurse, clients in best_solution.items():
    updated_clients = []
    for client in clients:
        if client not in mpp:
            updated_clients.append(client)
            mpp.add(client)
    best_solution[nurse] = updated_clients
nurse_schedules = {i: [0] * 2000 for i in range(1, nbr_of_nurse + 1)}
map=[]
for x,y in best_solution.items() :
  for i in y :
    h=find_start_time(x,i,time_spent,nurse_schedules[x])
    mark_nurse_busy(x,h,time_spent,nurse_schedules[x])
    map.append(i)

for j in range(1,nbr_client+1) :
  if j not in map :
    for x,y in best_solution.items() :
      upd=best_solution[x]
      h=find_start_time(x,j-1,time_spent,nurse_schedules[x])
      if h!=None :
        mark_nurse_busy(x,h,time_spent,nurse_schedules[x])
        upd.append(j)
        break
for x,y in best_solution.items():
  upd=best_solution[x]
  upd.sort(key=lambda j: df['DUE DATE'][j-1])
  best_solution[x]=upd

print("Best Solution:", best_solution)
nbr_srvd_clients = 0

for x, y in best_solution.items():
    nbr_srvd_clients += len(y)
print("number of served clients = ", nbr_srvd_clients)
total_dis = 0
for x, y in best_solution.items():
    total_dis += calculate_the_distance(y)
print("total_distance = ", total_dis)





import pandas as pd
import matplotlib.pyplot as plt
# Plotting the routes using matplotlib
plt.figure(figsize=(6, 4))

# Define the starting point
start_x, start_y = 40, 50

# Colors for different nurses
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k','r','k','b']
tr=0
for nurse, clients in best_solution.items():
    # Starting point (0, 0)
    route_x = [start_x]
    route_y = [start_y]
    tr+=1
    if tr<9 :
      continue
    if tr==10 :
      break
    # Get the coordinates of each client
    for client in clients:
        route_x.append(df.loc[df['CUST NO.'] == client, 'XCOORD.'].values[0])
        route_y.append(df.loc[df['CUST NO.'] == client, 'YCOORD.'].values[0])

    # Append the starting point to complete the route
    route_x.append(start_x)
    route_y.append(start_y)
    # Plot the route
    plt.plot(route_x, route_y, marker='o', color=colors[nurse % len(colors)], label=f'Nurse {nurse}')
# Plot the starting point
plt.scatter(start_x, start_y, color='k', s=100, label='Start (0, 0)')

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Nurse Routes from (40,50)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
