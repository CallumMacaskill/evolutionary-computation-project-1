import random as random
import numpy as numpy
import matplotlib.pyplot as plt

class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

class Individual:
    def __init__(self, bit_string):
        self.bit_string = bit_string
        self.probability = 0
        self.fitness = 0
        self.offset_fitness = 0

# Set up variables
files = ['data/knapsack-data/10_269', 'data/knapsack-data/23_10000', 'data/knapsack-data/100_1000']
optimal_values = [295, 9767, 1514]
bag_capacity = 0
optimum_value = 0
mutation_prob = 0
elitism = 0
population_size = 0
generations = 0
alpha = 0
items = []
lines = []
population = []
convergence_data = {}

def load_file(file_path):
    with open(file_path, 'r') as file:
        global optimum_value, mutation_prob, elitism, population_size, generations, alpha, bag_capacity
        if file_path == 'data/knapsack-data/10_269':
            optimum_value = 295
            mutation_prob = 0.2
            elitism = 5
            population_size = 50
            generations = 20
            alpha = 5
            if file_path not in convergence_data:
                convergence_data[file_path] = [[0 for col in range(5)] for row in range(generations)]
        elif file_path == 'data/knapsack-data/23_10000':
            optimum_value = 9767
            mutation_prob = 0.2
            elitism = 5
            population_size = 100
            generations = 60
            alpha = 5
            if file_path not in convergence_data:
                convergence_data[file_path] = [[0 for col in range(5)] for row in range(generations)]
        elif file_path == 'data/knapsack-data/100_1000':
            optimum_value = 1514
            mutation_prob = 0.2
            elitism = 25
            population_size = 500
            generations = 150
            alpha = 100
            if file_path not in convergence_data:
                convergence_data[file_path] = [[0 for col in range(5)] for row in range(generations)]
        lines = file.readlines()
    # Read the lines and create Item objects
    for x in range(len(lines)):
        tokens = lines[x].split(' ')
        if x == 0:
            bag_capacity = int(tokens[1])
        else:
            items.append(Item(int(tokens[0]), int(tokens[1])))

def initialise_population():
    for x in range(population_size):
        bit_string = []
        for y in range(len(items)):
            bit_string.append(str(random.randrange(2)))
        bit_string = ''.join(bit_string)
        population.append(Individual(bit_string))

def roulette_wheel():
    # Used the following link for implementing roulette wheel selection
    # https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
    # Find parent A
    fitness_sum = sum(individual.offset_fitness for individual in population)
    random_number = random.uniform(0, fitness_sum - 1)
    interval = 0
    for individual in population:
        interval += individual.offset_fitness
        if interval > random_number:
            parent = individual
            break
    return parent

def crossover(parent_a, parent_b):
    # 1-point crossover
    index = random.randrange(1, len(parent_a.bit_string))
    offspring_a_string = parent_a.bit_string[:index] + parent_b.bit_string[index:]
    offspring_b_string = parent_b.bit_string[:index] + parent_a.bit_string[index:]
    # Mutation
    prob_outcome = random.uniform(0, 1)
    if(prob_outcome < mutation_prob):
        # Perform mutation
        index = random.randrange(len(parent_a.bit_string))
        offspring_a_string = list(offspring_a_string)
        offspring_b_string = list(offspring_b_string)
        if offspring_a_string[index] == '0':
            offspring_a_string[index] = '1'
        else:
            offspring_a_string[index] = '0'
        if offspring_b_string[index] == '0':
            offspring_b_string[index] = '1'
        else: 
            offspring_b_string[index] = '0'
        offspring_a_string = ''.join(offspring_a_string)
        offspring_b_string = ''.join(offspring_b_string)
    return Individual(offspring_a_string), Individual(offspring_b_string)

def fitness_function(individual):
    individual_weight = 0
    individual_value = 0
    for x in range(len(individual.bit_string)):
        if(individual.bit_string[x] == "1"):
            individual_weight += items[x].weight
            individual_value += items[x].value
    return individual_value - (alpha * max(0, individual_weight - bag_capacity))

def genetic_algorithm(file_path):
    generation = 0
    global fitness_sum, population
    while generation < generations:
        # Update current population fitness values
        for individual in population:
            individual.fitness = fitness_function(individual)
        # Sort individuals based on their fitness score
        population.sort(key=lambda individual: individual.fitness, reverse=True)
        # Create new population
        new_population = []
        # Perform elitism
        for x in range(elitism):
            new_population.append(population[x])
        # Normalise fitness values to a positive range
        absolute_min = abs(min([individual.fitness for individual in population]))
        for individual in population:
            individual.offset_fitness = individual.fitness + absolute_min
        fitness_sum = sum(individual.offset_fitness for individual in population)
        population.sort(key=lambda individual: individual.offset_fitness, reverse=False)
        # Fill up new population
        while len(new_population) < population_size:
            parent_a = roulette_wheel()
            parent_b = roulette_wheel()
            offspring_a, offspring_b = crossover(parent_a, parent_b)
            new_population.append(offspring_a)
            new_population.append(offspring_b)
        # Record data for convergence graphs
        population.sort(key=lambda individual: individual.fitness, reverse=True)
        if population[0].fitness >= min(convergence_data[file_path][generation]):
            convergence_data[file_path][generation][convergence_data[file_path][generation].index(min(convergence_data[file_path][generation]))] = population[0].fitness
        population = new_population
        generation += 1

# Run genetic algorithm
best_solution_values = []
for dataset in files:
    dataset_solution_values = []
    for run_num in range(5):
        items.clear()
        lines.clear()
        population.clear()
        load_file(dataset)
        initialise_population()
        genetic_algorithm(dataset)
        # Calculate value of best solution from GA run
        population.sort(key=lambda individual: individual.fitness, reverse=True)
        best_weight = 0
        best_value = 0
        for x in range(len(population[0].bit_string)):
            if(population[0].bit_string[x] == "1"):
                best_weight += items[x].weight
                best_value += items[x].value
        if best_weight > bag_capacity:
            print("!!! EXCEEDED BAG CAPACITY !!!")
            print("on dataset", dataset)
        dataset_solution_values.append(best_value)
    best_solution_values.append(dataset_solution_values)

# Show mean and standard deviation of best solution values across 5 runs on each dataset
print("Mean values of best solutions across 5 runs:")
for x in range(len(files)):
    print(files[x], "=", numpy.mean(best_solution_values[x]), "       Optimal value =", optimal_values[x])
print("\nStandard deviation of best solutions across 5 runs:")
for x in range(len(files)):
    print(files[x], "=", numpy.std(best_solution_values[x]))
print("Done!")

# Plot convergence graphs
for count, dataset in enumerate(files):
    generations = []
    average_fitnesses = []
    for generation in range(len(convergence_data[dataset])):
        generations.append(generation)
        average_fitnesses.append(sum(convergence_data[dataset][generation])/len(convergence_data[dataset][generation]))
    plt.plot(generations, average_fitnesses)
    plt.suptitle(dataset + " convergence curve")
    plt.axhline(y=optimal_values[count], color='r', linestyle='-')
    plt.grid(True)
    plt.show()