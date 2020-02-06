
# coding: utf-8

# In[1]:


import random

from deap import base, creator, tools
creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)


# In[2]:


import re

def process_line(line):
    l = []
    for i in re.split('\s',line):
        if(i != ''):
            l.append(int(i))
    return l

def test_case(f_name = 'gk/gk01.dat'):
    f = open(f_name,"r")

    if(f == None):
        print("Invalid filename")

    line = f.readline()
    l = process_line(line)

    no_of_objects = l[0]
    no_of_constraints = l[1]
    optimum_value = l[2]

    print("Objects: ",no_of_objects)
    print("Constraints: ",no_of_constraints)
    print("Optimum value: ",optimum_value)

    profits = []
    count_n = 0
    while(count_n < no_of_objects):
        line = f.readline()
        l = process_line(line)
        profits.extend(l)
        count_n += len(l)

    print("Profits: ",len(profits))

    constraints = []
    count_m = 0
    count_n = 0

    while(count_m < no_of_constraints):
        constraints.append([])
        while(count_n < no_of_objects):
            line = f.readline()
            l = process_line(line)
            constraints[count_m].extend(l)
            count_n += len(l)		
        count_n = 0
        count_m = count_m + 1

    print("Constraints: ",len(constraints)*len(constraints[0]))

    capacity = []
    count_m = 0
    while(count_m < no_of_constraints):
        line = f.readline()
        l = process_line(line)
        capacity.extend(l)
        count_m += len(l)

    print("Capacity: ",len(capacity))
    
    requirements = [list(range(no_of_constraints)) for i in range(no_of_objects)]
    for i in range(no_of_objects):
        for j in range(no_of_constraints):
            requirements[i][j] = constraints[j][i]

    return (no_of_objects, no_of_constraints, optimum_value, profits, requirements, capacity)


# In[3]:


n, _,_, prof, requirements, capacity =  test_case("gk/gk01.dat")
'''n=4
prof = [70,80,90,200]
requirements = [[20], [30], [40],[70]]
capacity = [60]'''

def fitnessfunc(gene):
    fval = 0
    for i in range(len(gene)):
        if gene[i]==1:
            fval+=prof[i]
    totcap = [0 for i in capacity]
    numover = 0
    for i in range(len(gene)):
        for j in range(len(capacity)):
            if gene[i]==1:
                totcap[j]+=requirements[i][j]
    for i in range(len(capacity)):
        if totcap[i] > capacity[i]:
            numover+=1
    fval-= (max(prof)*numover*2)
    return (fval,)

tbx = base.Toolbox()

INDIVIDUAL_SIZE = n
import numpy as np
def genrandom(a, b):
    return int(np.random.choice([0,1],p= [0.6, 0.4]))

tbx.register("attr_int", random.randint, 0, 1)
tbx.register("individual",
            tools.initRepeat,
            creator.Individual,
            tbx.attr_int,
            n=INDIVIDUAL_SIZE)
tbx.register("population", tools.initRepeat, list, tbx.individual)
tbx.register("evaluate", fitnessfunc)
tbx.register("mate", tools.cxOnePoint)
tbx.register("mutate", tools.mutFlipBit, indpb = 0.1)
tbx.register("select", tools.selRoulette)


# In[30]:


def performGA(num, toolbox, CXPB, MUTPB, NGEN):
    pop = toolbox.population(n=num)

    # Evaluate the population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # perform roulette wheel selection
        offspring = toolbox.select(pop, len(pop))
        # clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    if(len(pop)==0):
        print("invalid population!!")
        return performGA(num, toolbox, CXPB, MUTPB, NGEN)
    return pop


# In[31]:


final_population = performGA(50, tbx, 0.6, 1/INDIVIDUAL_SIZE, 20)
fitnessfunc(sorted(final_population, reverse = True, key = fitnessfunc)[0])


# In[33]:


lst = []
from tqdm import tqdm
for i in tqdm(range(100)):
    population = tbx.population(n=50)
    from deap.algorithms import eaSimple
    final_population = performGA(50, tbx, 0.6, 1/INDIVIDUAL_SIZE, 200)
    final_population = list(sorted(final_population, key = fitnessfunc, reverse = True))
    lst.append(fitnessfunc(final_population[0]))
sorted(lst, reverse=True)

