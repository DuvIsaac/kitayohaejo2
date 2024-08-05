from Structures import *
from copy import deepcopy
from Flow import *


    ## Evaluation

def globalWaitingTime(network):
    matrix = [[0 for _ in network.shapes] for _ in network.stations]
    
    for s in network.stations:
        for i in range(len(network.shapes)):
            matrix[s.idt][i] = s.durations[i]
            for (lineNb, _) in s.paths[i]:
                matrix[s.idt][i] += network.lines[lineNb].waitingTime(network)
    
    mean = sum([sum(l) for l in matrix]) / len(network.stations)
    return mean * np.log(sum([sum([train.capacity for train in line.trains]) for line in network.lines])) * np.log(len(network.lines) + 1)


def meanWaitingTime(network):
    matrix = [[0 for _ in network.shapes] for _ in network.stations]
    
    for s in network.stations:
        for i in range(len(network.shapes)):
            matrix[s.idt][i] = s.durations[i]
            for (lineNb, _) in s.paths[i]:
                matrix[s.idt][i] += network.lines[lineNb].waitingTime(network)
    
    mean = sum([sum(l) for l in matrix]) / (len(network.stations) * len(network.shapes))
    return mean


    ## Mutations

def insertStation(network, line):
    available = [s.idt for s in network.stations if not s.idt in line.route]
    if len(available) != 0:
        t = rnd.choice(available)
        i = rnd.randint(0, len(line.route))
        line.route = line.route[: i] + [t] + line.route[i :]


def removeStation(network, line):
    i = rnd.randint(0, len(line.route) - 1)
    s = network.stations[line.route[i]]
    if len(s.lines) > 1:
        del line.route[i]
        if len(line.route) <= 1:
            del network.lines[line.nb]
            for i in range(line.nb, len(network.lines)):
                network.lines[i].nb -= 1


def insertTrain(network, line):
    train = Train(line.nb, 0, 0, [], 6)
    line.trains.append(train)


def changeCapacity(network, line):
    i = rnd.randint(0, len(line.trains) - 1)
    train = line.trains[i]
    newCapacity = train.capacity + rnd.choice([-1, 1]) * 6
    if newCapacity == 0 and len(line.trains) > 1:
        del line.trains[i]
    elif newCapacity != 0:
        train.capacity = newCapacity


def fusionPossible(lineA, lineB):
    if lineA.route[0] == lineB.route[0]:
        return (True, 0, 0)
    elif lineA.route[0] == lineB.route[-1]:
        return (True, 0, -1)
    elif lineA.route[-1] == lineB.route[0]:
        return (True, -1, 0)
    elif lineA.route[-1] == lineB.route[-1]:
        return (True, -1, -1)
    else:
        return (False, 1, 1)


def clean(routeFus):
    cleanRoute = []
    for s in routeFus:
        if not s in cleanRoute:
            cleanRoute.append(s)
    return cleanRoute


def fusionRoutes(routeA, routeB, extrA, extrB):
    if extrA == 0:
        return fusionRoutes(routeA[: : -1], routeB, -1, extrB)
    elif extrB == -1:
        return fusionRoutes(routeA, routeB[: : -1], extrA, 0)
    else:
        a, b = len(routeA), len(routeB)
        end = 0
        l = [i for i in range(min(a, b)) if routeA[a - 1 - i] != routeB[i]]
        if len(l) != 0:
            end = min(l)
        routeFus = routeA + routeB[end : :]
        routeFus = clean(routeFus)
        return routeFus


def fusion(network, lineA, lineB, extrA, extrB):

    routeFus = fusionRoutes(lineA.route, lineB.route, extrA, extrB)
    lineFus = Line(min(lineA.nb, lineB.nb), routeFus, lineA.trains + lineB.trains, False)

    network.lines[min(lineA.nb, lineB.nb)] = lineFus
    del network.lines[max(lineA.nb, lineB.nb)]
    for i in range(max(lineA.nb, lineB.nb), len(network.lines)):
        network.lines[i].nb -= 1


def COPossible(lineA, lineB):
    for i in range(len(lineA.route)):
        for j in range(len(lineB.route)):
            if lineA.route[i] == lineB.route[j]:
                return (True, i, j)
    return (False, -1, -1)


def CORoutes(routeA, routeB, startA, startB):
    firstA, endA = routeA[:startA], routeA[startA :]
    firstB, endB = routeB[:startB], routeB[startB :]
    return (clean(firstA + endB), clean(firstB + endA))


def crossOverRoutes(network, lineA, lineB, startA, startB):
    (routeAB, routeBA) = CORoutes(lineA.route, lineB.route, startA, startB)
    network.lines[lineA.nb].route = routeAB
    network.lines[lineB.nb].route = routeBA


def crossOverLines(network, lineA, lineB, p):
    (b, extrA, extrB) = fusionPossible(lineA, lineB)
    (c, startA, startB) = COPossible(lineA, lineB)
    if b and rnd.random() < p:
            fusion(network, lineA, lineB, extrA, extrB)
            network.updateAllPaths()
    elif c and rnd.random() < p:
            crossOverRoutes(network, lineA, lineB, startA, startB)
            network.updateAllPaths()


def swap(L, i, k):

    if i > k:
        return swap(L, k, i)
    
    j = (i + 1) % len(L)
    l = (k + 1) % len(L)

    r1 = L[: i + 1]
    r2 = L[j : k + 1]
    r3 = L[k + 1 :]

    return r1 + r2[:: -1] + r3


def OPT2(network, lineNb):
    line = network.lines[lineNb]
    bestRoute = line.route
    bestWeight = globalWaitingTime(network)
    bestNetwork = network
    i = 0

    while i < len(bestRoute):
        j = i + 1

        while j < len(bestRoute):
            newR = swap(bestRoute, i, j)
            newNet = deepcopy(bestNetwork)
            newNet.lines[lineNb].route = newR
            newNet.updateAllPaths()
            newWeight = globalWaitingTime(newNet)

            if newWeight < bestWeight:
                bestNetwork = newNet
                bestRoute = newR
                bestWeight = newWeight
                i = 0
                j = 0
            else:
                j += 1
        
        i += 1
    print('AH')


def crossOver(network, p):
    lineA = rnd.choice(network.lines)
    lineB = rnd.choice(network.lines)
    if lineA.nb != lineB.nb:
        crossOverLines(network, lineA, lineB, p)


def mutate(network):
    p = rnd.random()
    line = rnd.choice(network.lines)
    if p < 0.05:
        insertStation(network, line)
    elif p < 0.1:
        removeStation(network, line)
    elif p < 0.15:
        insertTrain(network, line)
    elif p < 0.2:
        changeCapacity(network, line)
    else:
        crossOver(network, 0.9)
    network.updateAllPaths()


    ## Main algorithm

def startSample(network, n):
    population = [deepcopy(network) for _ in range(n)]
    for indiv in population:
        for _ in range(10):
            mutate(indiv)
    return population


def geneticMaybe(network):
    population = startSample(network, 10)
    nextGen = deepcopy(population)
    for i in range(100):
        print(str(i) + '%')
        if len(population) <= 3:
            population += deepcopy(population)
            population += deepcopy(population)
        for indiv in population:
            new = deepcopy(indiv)
            mutate(new)
            if len(new.lines) > 0:
                nextGen.append(new)
        nextGen.sort(key=globalWaitingTime)
        print(globalWaitingTime(nextGen[0]))
        population = nextGen[: min(10, len(population))]
    net = sorted(population, key=globalWaitingTime)[0]
    return net


def default(n, p):
    net = randomEmptyNetwork(n, p)
    exhaustEdges(net, monotoneSelector)
    return net


def solve(n, p):
    net = default(n, p)
    return geneticMaybe(net)




    







