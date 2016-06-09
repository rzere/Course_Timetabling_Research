#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools
from solver import schedule_ctb_file_with_parameters

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

NUM_TRAINING_FILES = 10
NUM_TESTING_FILES = 0

toolbox = base.Toolbox()

# Output log file
f = open('one_pass_genetic_optimization_log.txt', 'w')

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_5_to_50_by_1", random.randint, 5, 50)
toolbox.register("attr_0_to_20_by_1", random.randint, 0, 20)
toolbox.register("attr_0_to_50_by_1", random.randint, 0, 50)
toolbox.register("attr_5_to_50_by_5", random.randint, 1, 10)
toolbox.register("attr_100_to_2000_by_50", random.randint, 2, 40)


# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#    toolbox.attr_int, 4)

toolbox.register("individual", tools.initCycle, creator.Individual,
    (toolbox.attr_5_to_50_by_1, toolbox.attr_0_to_20_by_1, toolbox.attr_0_to_50_by_1, toolbox.attr_5_to_50_by_5, toolbox.attr_100_to_2000_by_50), n = 1)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def onePassSolver(individual):

    parameters = []
    parameters.append(individual[0])
    parameters.append(individual[1])
    parameters.append(individual[2])
    parameters.append((individual[3] + 1) * 5)
    parameters.append((individual[4] + 2) * 50)

    # Open the log file
    base_dir = '/Users/ruzfl/Documents/Dropbox/DanJayRuz/Test_Problems_3_to_7_Choices/'
    file_base = 'Course_Schedules_Fall_2015_'

    total_penalty = 0.0

    f.write('Individual ' + str(parameters) + '\n')
    f.write('Training:\n')

    # Run the 10 training problems
    total_training_penalty = 0

    for file_number in range(NUM_TRAINING_FILES):
        file_name = file_base + str(file_number) + '.ctb'
        file_path = base_dir + file_name
        penalty = schedule_ctb_file_with_parameters(parameters, file_path)
        total_training_penalty += penalty

        # Log to the output file
        f.write('    ' + str(parameters) + '    ' + file_name + '    ' + str(penalty) + '\n')
        print str(parameters) + '    ' + file_name + '    ' + str(penalty)

    f.write('    Average training penalty: ' + str(total_training_penalty / NUM_TRAINING_FILES) + '\n')

    # Run the five test problems
    #f.write('Testing:\n')

    #total_testing_penalty = 0

    #for file_number in range(NUM_TESTING_FILES):
    #    file_name = file_base + str(file_number + NUM_TRAINING_FILES) + '.ctb'
    #    file_path = base_dir + file_name
    #    penalty = schedule_ctb_file_with_parameters(parameters, file_path)
    #    total_testing_penalty += penalty

    #    # Log to the output file
    #    f.write('    ' + str(parameters) + '    ' + file_name + '    ' + str(penalty) + '\n')
    #    print str(parameters) + '    ' + file_name + '    ' + str(penalty)

    #f.write('    Average test penalty: ' + str(total_testing_penalty / NUM_TESTING_FILES) + '\n')

    f.flush()

    average_training_penalty = total_training_penalty / NUM_TRAINING_FILES
    return (average_training_penalty,)


#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", onePassSolver)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=100)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 20

    f.write("Start of evolution\n")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    f.write("  Evaluated %i individuals\n" % len(pop))
    best_ind = tools.selBest(pop, 1)[0]
    f.write("Best individual is %s, %s\n" % (best_ind, best_ind.fitness.values))


    # Begin the evolution
    for g in range(NGEN):
        f.write("-- Generation %i --\n" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        f.write("  Evaluated %i individuals\n" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        f.write("  Min %s\n" % min(fits))
        f.write("  Max %s\n" % max(fits))
        f.write("  Avg %s\n" % mean)
        f.write("  Std %s\n" % std)
        best_ind = tools.selBest(pop, 1)[0]
        f.write("Best individual is %s, %s\n" % (best_ind, best_ind.fitness.values))


    f.write("-- End of (successful) evolution --\n")

    best_ind = tools.selBest(pop, 1)[0]
    f.write("Best individual is %s, %s\n" % (best_ind, best_ind.fitness.values))

    f.close()

if __name__ == "__main__":
    main()
