# python packages
import random
import time
import operator

import sklearn
import evalGP_main as evalGP
# only for strongly typed GP
import gp_restrict
import numpy as np
# deap package
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Img, Region, Vector
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

'FLGP'

dataSetName='f2'
randomSeeds=2

x_train = np.load('data/FEI-dataset/' + dataSetName + '/' + dataSetName + '_train_data.npy') / 255.0
y_train = np.load('data/FEI-dataset/' + dataSetName + '/' + dataSetName + '_train_label.npy')
x_test = np.load('data/FEI-dataset/' + dataSetName + '/' + dataSetName + '_test_data.npy') / 255.0
y_test = np.load('data/FEI-dataset/' + dataSetName + '/' + dataSetName + '_test_label.npy')

print(x_train.shape)

# parameters:
population = 10
generation = 10
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8

bound1, bound2 = x_train[1, :, :].shape
##GP

pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
#Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='FeaCon2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector, name='FeaCon3')
# Global feature extraction
pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
# Local feature extraction
pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
# Region detection operators
pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
# Terminals
pset.renameArguments(ARG0='Grey')
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 51), Int3)

#fitnesse evaluaiton
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)

def evalTrain(individual):
    # print(individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    # print(train_norm.shape)
    lsvm = LinearSVC(max_iter=100)
    accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
    return accuracy,

def evalTrainb(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i, :, :])))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm = LinearSVC(max_iter=100)
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
    except:
        accuracy = 0
    return accuracy,

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof

def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    for j in range(0, len(y_test)):
        test_tf.append(np.asarray(func(x_test[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC()
    lsvm.fit(train_norm, y_train)
    accuracy = round(100*lsvm.score(test_norm, y_test),2)
    return train_tf.shape[1], accuracy

# MY CODE BELOW ---------------------------

def write_features_csv(function, instances, labels, type):
    # Extract features from instances using expression found from GP
    csv_instances = []
    for count, instance in enumerate(instances):
        csv_instances.append(np.append(function(instance), labels[count]))
    # Write features to CSV
    file = open(dataSetName + '_automatic_' + type + '_tabular_dataset.csv', 'w', newline='\n')
    writer = csv.writer(file, delimiter=',')
    writer.writerows(csv_instances)
    file.close()

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

def feature_label_split(instances):
    # Extract features and labels
    features = []
    labels = []
    for instance in instances:
        features.append(instance[:-1])
        labels.append(instance[-1])
    return features, labels

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    num_features, testResults = evalTest(hof[0])
    endTime1 = time.process_time()
    testTime = endTime1 - endTime

    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')

    # 4.4
    # Get the best performing individual and write CSV
    best_function = toolbox.compile(expr=hof[0])
    write_features_csv(best_function, x_train, y_train, "training")
    write_features_csv(best_function, x_test, y_test, "testing")
    
    # 4.5
    # Test automatic on NB
    training_instances = read_csv(dataSetName + '_automatic_training_tabular_dataset.csv')
    testing_instances = read_csv(dataSetName + '_automatic_testing_tabular_dataset.csv')

    # Collect TPR/TNR data
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Extract features and labels
    training_features, training_labels = feature_label_split(training_instances)
    testing_features, testing_labels = feature_label_split(testing_instances)

    # Train and test on Naive Bayes
    classifier = GaussianNB()
    classifier.fit(training_features, training_labels)
    classification_accuracy = classifier.score(training_features, training_labels)

    # Find TPR
    for count, instance in enumerate(training_features):
        label = training_labels[count]
        prediction = int(classifier.predict([instance])[0])
        if label == 1 and prediction == 1:
            true_positives += 1
        elif label == 0 and prediction == 1:
            false_positives += 1
        elif label == 0 and prediction == 0:
            true_negatives += 1
        elif label == 1 and prediction == 0:
            false_negatives += 1
    tpr = true_positives/(true_positives + false_negatives)
    fpr = false_positives/(false_positives + true_negatives)
    print("Automatic feature selection")
    print(dataSetName, "   Training Classification Accuracy:", classification_accuracy, "    TPR:", tpr, "    FPR:", fpr)

    # ROC graphs
    plot_roc_curve(classifier, training_features, training_labels)
    plt.suptitle(dataSetName + " Training ROC Curve")
    plt.show()

    # Train and test on Naive Bayes
    classification_accuracy = classifier.score(testing_features, testing_labels)

    # Find TPR
    for count, instance in enumerate(testing_features):
        label = testing_labels[count]
        prediction = int(classifier.predict([instance])[0])
        if label == 1 and prediction == 1:
            true_positives += 1
        elif label == 0 and prediction == 1:
            false_positives += 1
        elif label == 0 and prediction == 0:
            true_negatives += 1
        elif label == 1 and prediction == 0:
            false_negatives += 1
    tpr = true_positives/(true_positives + false_negatives)
    fpr = false_positives/(false_positives + true_negatives)
    print("Automatic feature selection")
    print(dataSetName, "   Testing Classification Accuracy:", classification_accuracy, "    TPR:", tpr, "    FPR:", fpr)

    # ROC graphs
    plot_roc_curve(classifier, testing_features, testing_labels)
    plt.suptitle(dataSetName + " Testing ROC Curve")
    plt.show()

    # Compare with manually selected features
    training_instances = read_csv(dataSetName + '_manual_training_tabular_dataset.csv')
    testing_instances = read_csv(dataSetName + '_manual_testing_tabular_dataset.csv')
    training_features, training_labels = feature_label_split(training_instances)
    testing_features, testing_labels = feature_label_split(testing_instances)
    classifier = GaussianNB()
    classifier.fit(training_features, training_labels)
    classification_accuracy = classifier.score(testing_features, testing_labels)
    print("Manual feature selection")
    print(dataSetName, "   Test Classification Accuracy:", classification_accuracy)