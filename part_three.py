import operator
import math
from deap import gp, creator, base, tools, algorithms
from numpy import exp

# Read Part 1 and Part 2 of the basic tutorials and also the Genetic Programming advanced tutorials from the resource provided in the handout
# https://deap.readthedocs.io/en/master/

# Used code from the genetic programming example from the same resource
# https://deap.readthedocs.io/en/master/examples/gp_symbreg.html

# Specially designed division to avoid 0 error
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def square(input):
    return input**2

def is_positive(input):
    return True if input > 0 else False

def if_then_else(input, output1, output2):
    return output1 if input else output2

# Creating the function set for tree nodes
pset = gp.PrimitiveSetTyped("MAIN", [float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(square, [float], float)
pset.addPrimitive(is_positive, [float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# Creating the terminal set for tree nodes
for x in range(6):
    pset.addTerminal(x + 1, float)
pset.addTerminal(1, bool)
pset.addTerminal(0, bool)

# Create the types for this GP problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Negative weights because this is a minimisation problem
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Create tools for this evolution process
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Find what the expected result using the target formula should be for a given input
def calculate_real_value(input):
    if input > 0:
        return (1 / input) + math.sin(input)
    else:
        return 2 * input + math.pow(input, 2) + 3.0

# Create the fitness function that individuals are evaluated against
def fitness_function(individual, fitness_cases):
    # Transform the Individual's tree into a callable expression
    function = toolbox.compile(expr=individual)
    tree = gp.PrimitiveTree(individual)
    # Evaluate the MSE between the expression and real function value
    square_errors = []
    for case in fitness_cases:
        expression_value = function(case)
        real_value = calculate_real_value(case)
        square_errors.append((expression_value - real_value)**2)
    return sum(square_errors)/len(square_errors),

# Used this forum thread to help create my selection method implementing tournament selection and also elitism
# https://groups.google.com/g/deap-users/c/iannnLI2ncE
def select_tournament_elitism(individuals, pop_size):
    return tools.selBest(individuals=individuals, k=int(0.1*pop_size)) + tools.selTournament(individuals=individuals, k=pop_size - int(0.1*pop_size), tournsize=7)

# Add more tools for the GP to use in the evolution process#
toolbox.register("evaluate", fitness_function, fitness_cases=[x for x in range(-100, 100)])
toolbox.register("select", select_tournament_elitism)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Begin the evolution process
population = toolbox.population(n=3000)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(population, toolbox, 0.5, 0.1, 100, None, hof, False)

def print_results(function):
    for x in range(-50, 50):
        print(x * 2,") Expected:", calculate_real_value(x * 2), "     GP Result:", function(x * 2))

best_function = toolbox.compile(expr=hof[0])
tree = gp.PrimitiveTree(hof[0])
print(str(tree))
print_results(best_function)
print("Done!")