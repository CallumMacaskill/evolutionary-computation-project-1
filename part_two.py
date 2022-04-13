from itertools import tee
import random as random
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Individual:
    def __init__(self, bit_string):
        self.bit_string = bit_string
        self.fitness = -1

files = ['data/part-two-data/wbcd.data', 'data/part-two-data/sonar.data']
instances = []
filter_instances = []
labels = []
population_size = 0
generations = 0
population = []
mutation_prob = 0.0
elitism = 0
fitness_sum = 0

def load_file(file_path, type):
    global generations, population_size, mutation_prob, elitism, filter_instances, instances
    instances.clear()
    labels.clear()
    with open(file_path, 'r') as file:
        lines = file.readlines()
    if file_path == files[0]:
        if type == "wrapper":
            generations = 15
            population_size = 100
            mutation_prob = 0.2
            elitism = 5
        elif type == "filter":
            generations = 15
            population_size = 100
            mutation_prob = 0.2
            elitism = 5
    elif file_path == files[1]:
        if type == "wrapper":
            generations = 15
            population_size = 100
            mutation_prob = 0.2
            elitism = 50
        elif type == "filter":
            generations = 15
            population_size = 100
            mutation_prob = 0.2
            elitism = 20
    # Read lines for tokens and add to instance lists
    for x in range(len(lines)):
        instance = lines[x].split(',')
        label = int(instance.pop())
        instances.append(instance)
        labels.append(label)
    # Perform discretisation transformation. Learned about it and used code from the following links
    # https://machinelearningmastery.com/information-gain-and-mutual-information/
    # https://machinelearningmastery.com/discretization-transforms-for-machine-learning/
    kbins = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    filter_instances = kbins.fit_transform(instances)

def initialise_population():
    population.clear()
    for x in range(population_size):
        bit_string = []
        for y in range(len(instances[0])):
            bit_string.append(str(random.randrange(2)))
        bit_string = ''.join(bit_string)
        population.append(Individual(bit_string))

def roulette_wheel():
    # Used the following link for implementing roulette wheel selection
    # https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
    # Find parent A
    fitness_sum = sum(individual.fitness for individual in population)
    random_number = random.uniform(0, fitness_sum - 1)
    interval = 0
    for individual in population:
        interval += individual.fitness
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

def wrapper_fitness_function(individual):
    bit_string = list(individual.bit_string)
    # Transform data to only contain features defined in GA bits 
    transformed_instances = []
    for instance_index in range(len(instances)):
        transformed_instance = []
        for bit_index in range(len(bit_string)):
            if bit_string[bit_index] == '1':
                transformed_instance.append(instances[instance_index][bit_index])
        transformed_instances.append(transformed_instance)
    # Score it
    X_train, X_test, y_train, y_test = train_test_split(transformed_instances, labels, test_size=.3, random_state=20)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def filter_fitness_function(individual):
    bit_string = list(individual.bit_string)
    # Transform data to only contain features defined in GA bits 
    transformed_instances = []
    for instance_index in range(len(filter_instances)):
        transformed_instance = []
        for bit_index in range(len(bit_string)):
            if bit_string[bit_index] == '1':
                transformed_instance.append(filter_instances[instance_index][bit_index])
        transformed_instances.append(transformed_instance)
    # Find the mutual information score. Found the following link handy for seeing SKLearn component in action.
    # https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8
    mi_score = mutual_info_classif(transformed_instances, labels)
    return sum(mi_score)/len(mi_score)

def naive_bayes(individual):
    bit_string = list(individual.bit_string)
    # Transform data to only contain features defined in GA bits 
    transformed_instances = []
    for instance_index in range(len(instances)):
        transformed_instance = []
        for bit_index in range(len(bit_string)):
            if bit_string[bit_index] == '1':
                transformed_instance.append(instances[instance_index][bit_index])
        transformed_instances.append(transformed_instance)
    # Create, train, and score a KNN classifier
    X_train, X_test, y_train, y_test = train_test_split(transformed_instances, labels, test_size=.3)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier.score(X_test, y_test)

# Followed pseudocode supplied in lecture slides
def genetic_algorithm(type):
    global generations, population, fitness_sum
    generation = 0
    while generation < generations:
        # Update current population fitness values
        for individual in population:
            if type == "wrapper":
                individual.fitness = wrapper_fitness_function(individual)
            elif type == "filter":
                individual.fitness = filter_fitness_function(individual)
        # Sort individuals based on their fitness score
        population.sort(key=lambda individual: individual.fitness, reverse=True)
        # Create new population
        new_population = []
        # Perform elitism
        for x in range(elitism):
            new_population.append(population[x])
        fitness_sum = sum(individual.fitness for individual in population)
        population.sort(key=lambda individual: individual.fitness, reverse=False)
        # Fill up new population
        while len(new_population) < population_size:
            parent_a = roulette_wheel()
            parent_b = roulette_wheel()
            offspring_a, offspring_b = crossover(parent_a, parent_b)
            new_population.append(offspring_a)
            new_population.append(offspring_b)
        population = new_population
        generation += 1

# Do a full run though of the program to parse data, create population, run GA, and get a best result.
def run():
    for x in range(2):
        file_number = x
        for y in range(2):
            if y == 0:
                type = "wrapper"
            elif y == 1:
                type = "filter"
            global population
            nb_scores = []
            computational_times = []
            # Repeat 5 times to get an average
            for z in range(5):
                load_file(files[file_number], type)
                initialise_population()
                start_time = time.time()
                genetic_algorithm(type)
                # Record computational time
                end_time = time.time()
                computational_times.append(end_time - start_time)
                # Choose what fitness function to apply
                for individual in population:
                    if type == "wrapper":
                        individual.fitness = wrapper_fitness_function(individual)
                    elif type == "filter":
                        individual.fitness = filter_fitness_function(individual)
                population.sort(key=lambda individual: individual.fitness, reverse=True)
                nb_scores.append(naive_bayes(population[0]))
                print("Completed iteration", z, "on file", x, "with type", type)
            nb_average = np.mean(nb_scores)
            nb_std = np.std(nb_scores)
            computational_time_average = np.mean(computational_times)
            computational_time_std = np.std(computational_times)

            # Get an average of the default Naive Bayes score using all of the features
            default_scores = []
            for x in range(5):
                default_scores.append(get_default_score(files[file_number]))
            default_average = np.mean(default_scores)
            default_std = np.std(default_scores)

            print("\nFile:                ", files[file_number])
            print("Fitness type:          ", type)
            print("Average GA score:      ", nb_average)
            print("GA std:                ", nb_std)
            print("Computation time mean: ", computational_time_average)
            print("Computational time std:", computational_time_std)
            print("")
            print("Average Default score: ", default_average)
            print("Default score std:     ", default_std)
            print("--------------------------------------------------------")
            

# Run instances with all of their features through the Naive Bayes classifier to get a default score
def get_default_score(file_path):
    default_instances = []
    default_labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for x in range(len(lines)):
        default_instance = lines[x].split(',')
        default_label = int(default_instance.pop())
        default_instances.append(default_instance)
        default_labels.append(default_label)
    X_train, X_test, y_train, y_test = train_test_split(default_instances, default_labels, test_size=.3, shuffle=True)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier.score(X_test, y_test)

run()