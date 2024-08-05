from Structures import *
from NetworkBuilder import randomEmptyNetwork


def weightBasic(network, branch):
    return sum([network.distances[branch[i]][branch[(i + 1) % len(branch)]] for i in range(len(branch))])


def score(network, s, branch):
    l = [min(abs(s - k), abs(s - len(branch) + k)) for k in range(1, len(branch)) if network.stations[branch[s]].shape == network.stations[branch[k]].shape]
    if len(l) == 0:
        return 0
    else:
        return len(branch) - min(l)


def evaluateDebit(network, branch):
    clusterNb = 20
    length = 0
    i = 0
    while i < len(branch):
        s1 = network.stations[branch[i]]
        s2 = network.stations[branch[(i + 1) % len(branch)]]
        length += network.distances[s1.idt][s2.idt]
        if s1.shape != s2.shape:
            clusterNb += 1
        i += 1
    return length / clusterNb


def totalWeight(network, branch):

    return evaluateDebit(network, branch) / len(set([network.stations[i].shape for i in branch]))


def minmax(L):
    m = max(L)
    return min([i for i in range(len(L)) if L[i] == m])


def listWaiting(network):
    return [np.ceil(100 / s.spTime) for s in network.stations]


def updateWaiting(network, waiting, line):
    i = 0
    while i < len(line.route):
        s2 = network.stations[line.route[i]]
        s1 = network.stations[line.route[(i - 1) % len(line.route)]]
        if s1.shape != s2.shape:
            waiting[line.route[i]] = max(0, waiting[line.route[i]] - 6)
        elif waiting[s1.idt] < 6:
            waiting[s2.idt] = max(0, waiting[s2.idt] - 6 + waiting[s1.idt])
            waiting[s1.idt] = 0
        i += 1
        

def oneLine(network, waiting, s, lineNb, totalLines=8):
    n = len(network.stations)
    deja_vu = [False for _ in range(n)]
    branch = [s]
    deja_vu[s] = True
    stop = False
    count = 0

    while not stop and count < len(network.stations):

        costs = [float('inf') for _ in range(n ** 2 + 2)]

        for k in range(n):
            if not deja_vu[k]:
                for i in range(n):
                    costs[n * k + i] = totalWeight(network, branch[: i] + [k] + branch[i :])
        
        if count >= len(waiting) / totalLines:
            costs[n ** 2] = totalWeight(network, branch + [s])
            costs[n ** 2 + 1] = totalWeight(network, branch + branch[n - 1 : 0 : -1]) * 3 / 4
        
        m = min(costs)
        i = min([j for j in range(len(costs)) if costs[j] == m])

        isCyclic = True

        if i < n ** 2:
            k = i // n
            j = i % n
            branch = branch[: j] + [k] + branch[j :]
            deja_vu[k] = True
            count += 1
        elif i == n:
            stop = True
        else:
            stop = True
            isCyclic = False
    
    line = Line(lineNb, branch, [], cyclic=isCyclic)

    updateWaiting(network, waiting, line)
        
    return line


def appendLast(network):

    for station in network.stations:

        d = [min([network.distances[station.idt][j] for j in line.route]) for line in network.lines]
        dmin = min(d)

        if dmin != 0:
            i = min([j for j in range(len(d)) if d[j] == dmin])
            line = network.lines[i]

            additionnalCosts = [network.distances[line.route[j]][station.idt] + network.distances[station.idt][line.route[(j + 1) % len(line.route)]] for j in range(len(line.route))]
            cmin = min(additionnalCosts)
            j = min([k for k in range(len(line.route)) if additionnalCosts[k] == cmin])
            line.route = line.route[: j + 1] + [station.idt] + line.route[j + 1 :] 



def glutton(network, totalLines=8):
    waiting = listWaiting(network)
    lineNb = 0

    while totalLines > lineNb:
        M = max(waiting)
        s = min([j for j in range(len(waiting)) if waiting[j] == M])
        line = oneLine(network, waiting, s, lineNb)
        network.addLine(line)
        lineNb += 1
    
    appendLast(network)
    network.updateAllPaths()



        


        
