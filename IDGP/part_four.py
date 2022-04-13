import numpy as np
import csv

from numpy.random import f
import feature_function as feature_functions
import gp_restrict
import evalGP_main
from deap import gp, creator, base, tools, algorithms
import operator
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# Use predefined regions to extract features from the images
def extract_features(images_path, labels_path):
    feature_instances = []
    images = np.load(images_path)
    image_labels = np.load(labels_path)
    for count, image in enumerate(images):
        # Global region's features
        global_features = feature_functions.all_sift(image)
        # Define local regions
        left_eye_region = feature_functions.regionS(image, 70, 30, 30)
        right_eye_region = feature_functions.regionS(image, 70, 80, 30)
        teeth_region = feature_functions.regionR(image, 125, 50, 15, 30)
        cheek_region = feature_functions.regionS(image, 110, 30, 20)
        # Get features from local regions
        left_eye_features = feature_functions.all_sift(left_eye_region)
        right_eye_features = feature_functions.all_sift(right_eye_region)
        teeth_features = feature_functions.all_sift(teeth_region)
        cheek_features = feature_functions.all_sift(cheek_region)
        # Get averages of sift features
        global_mean = np.mean(global_features)
        left_eye_mean = np.mean(left_eye_features)
        right_eye_mean = np.mean(right_eye_features)
        teeth_mean = np.mean(teeth_features)
        cheek_mean = np.mean(cheek_features)
        # Combine data
        image_features = np.concatenate((global_mean, left_eye_mean, right_eye_mean, teeth_mean, cheek_mean), axis=None)
        instance = np.append(image_features, image_labels[count])
        feature_instances.append(instance)
    return feature_instances

# Converts the given instances list into a CSV
# https://www.pythontutorial.net/python-basics/python-write-csv-file/
def write_csv(instances, type, file_num):
    file = open('f' + str(file_num) + '_manual_' + type + '_tabular_dataset.csv', 'w', newline='\n')
    writer = csv.writer(file, delimiter=',')
    writer.writerows(instances)
    file.close()

# https://realpython.com/python-csv/
def read_csv(file):
    instances = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            instance = []
            for token in row:
                instance.append(float(token))
            instances.append(instance)
    return instances

def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def fitness_function(individual, instances):
    function = toolbox.compile(expr=individual)
    probabilities = []
    correct_classifications = 0
    for instance in instances:
        features = instance[:-1]
        label = instance[-1]
        expression_value = function(*features)
        if expression_value >= 0 and label == 1:
            correct_classifications += 1
        elif expression_value < 0 and label == 0:
            correct_classifications += 1
        probabilities.append(function(*features))
    return correct_classifications/len(instances), probabilities

def select_elitism(population, k):
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    return population[:k]

def select_tournament_elitism(individuals, pop_size):
    return tools.selBest(individuals=individuals, k=int(0.1*pop_size)) + tools.selTournament(individuals=individuals, k=pop_size - int(0.1*pop_size), tournsize=7)

# Create the types for this GP problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

for file_num in range(1, 3):
    # Extract features from training and test images
    training_instances = extract_features("IDGP/f" + str(file_num) + "_train_data.npy", "IDGP/f" + str(file_num) + "_train_label.npy")
    testing_instances = extract_features("IDGP/f" + str(file_num) + "_test_data.npy", "IDGP/f" + str(file_num) + "_test_label.npy")
    # Create CSVs from feature instances
    write_csv(training_instances, "training", file_num)
    write_csv(testing_instances, "testing", file_num)

    # Get instances out of CSV
    train_instances = read_csv("f" + str(file_num) + "_manual_training_tabular_dataset.csv")
    test_instances = read_csv("f" + str(file_num) + "_manual_testing_tabular_dataset.csv")

    pset = gp.PrimitiveSet("main", len(training_instances[0]) - 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    for x in range(-10, 10):
        if x != 0:
            pset.addTerminal(x)

    # Create tools for this evolution process
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_function, instances=train_instances)
    toolbox.register("select", select_tournament_elitism)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # 4.3 -----------------------------
    # Initialise population
    initial_population = toolbox.population(n=3000)
    hof = tools.HallOfFame(1)
    population, logbook = algorithms.eaSimple(initial_population, toolbox, 0.5, 0.1, 50, None, hof, False)
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    print("Data set: F", file_num)
    train_accuracy, train_prob = fitness_function(population[0], train_instances)
    test_accuracy, test_prob = fitness_function(population[0], test_instances)
    print("Train data accuracy:", train_accuracy)
    print("Test data accuracy: ", test_accuracy)
    # ROC curve
    features = []
    labels = []
    for instance in test_instances:
        features.append(instance[:-1])
        labels.append(instance[-1])
    fpr, tpr, threshold = roc_curve(labels, test_prob)
    plt.plot(fpr, tpr)
    plt.suptitle("F" + str(file_num) + " ROC Curve")
    plt.show()
    print("Done!")