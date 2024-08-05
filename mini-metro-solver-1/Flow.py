from Structures import *
from scipy.spatial import Delaunay
from NetworkBuilder import *


def dijkstra(graph, statShapes, start, tabShapes):

    paths = []
    n = len(graph)

    spanningForest = [None for _ in range(n)]
    dejaVu = [False for _ in range(n)]
    U = PriorityQueue(n)
    closer = [None for _ in tabShapes]

    U.push(start, 0)
    
    while U.length() != 0 and len([x for x in closer if x is None]) != 0:
        
        (u, k) = U.pop()
        if not dejaVu[u]:
            dejaVu[u] = True

            for i in range(len(tabShapes)):
                if closer[i] is None and statShapes[u] == tabShapes[i]:
                    closer[i] = u

            for (v, w) in graph[u]:
                if - U.priority(v) > - k + w:
                    U.changePrio(v, k - w)
                    spanningForest[v] = u
    
    for i in range(len(tabShapes)):
        route = []
        goal = closer[i]

        while goal != start:
            route.append(goal)
            goal = spanningForest[goal]

        paths.append(route)

    return paths


def buildFlowRoute(graph, start, shapeNb, flowGraph, route, influx):
    current = start

    while len(route) != 0:
        dest = route.pop()
        flowGraph[current][dest] += 60 * influx[current][shapeNb]
        current = dest


def buildFlowGraph(graph, statShapes, influx, tabShapes):
    flowGraph = [[0 for _ in graph] for _ in graph]

    for vertex in range(len(graph)):
        paths = dijkstra(graph, statShapes, vertex, tabShapes)
        for i in range(len(tabShapes)):
            buildFlowRoute(graph, vertex, i, flowGraph, paths[i], influx)
    
    return flowGraph


def insert(graph, i, j, d):
    b = True
    for (k, _) in graph[i]:
        if k == j:
            b = False
    if b:
        graph[i].append((j, d))
        graph[j].append((i, d))


def initialGraph(network):
    points = np.array([[station.loc[0], station.loc[1]] for station in network.stations])
    triangulation = Delaunay(points).simplices
    graph = [[] for _ in network.stations]

    for [i, j, k] in triangulation:
        insert(graph, i, j, network.distances[i][j])
        insert(graph, i, k, network.distances[i][k])
        insert(graph, j, k, network.distances[j][k])
    
    return graph


def removeUseless(graph, flowGraph):
    for i in range(len(graph)):
        useful = []
        for (j, w) in graph[i]:
            if flowGraph[i][j] > 0.01:
                useful.append((j,w))
        graph[i] = useful


def selectFirst(flowGraph):
    k, s, i = len(flowGraph), float('inf'), -1

    for j in range(len(flowGraph)):
        v = [u for u in range(len(flowGraph)) if flowGraph[j][u] != 0]
        if 0 < len(v) < k:
            k = len(v)
            s = sum([f for f in flowGraph[j]])
            i = j
        elif len(v) == k and k != 0:
            t = sum([f for f in flowGraph[j]])
            if t < s:
                k, s, i = len(v), t, j
    
    return i


def monotoneSelector(flowGraph):
    branch = []

    s = selectFirst(flowGraph)
    branch.append(s)
    neighbours = [j for j in range(len(flowGraph)) if flowGraph[s][j] != 0 and not j in branch]

    while len(neighbours) > 0:
        M = max([flowGraph[s][k] for k in neighbours])
        t = [k for k in neighbours if flowGraph[s][k] == M][0]
        branch.append(t)
        s = t
        neighbours = [j for j in range(len(flowGraph)) if flowGraph[s][j] != 0 and not j in branch]

    return branch 


def estimateTrainCost(branch, network, flowGraph):
    fmin = min([flowGraph[branch[i]][branch[i + 1]] for i in range(len(branch) - 1)])
    d = sum([network.distances[branch[i]][branch[i + 1]] for i in range(len(branch) - 1)])
    cyclic = False
    servedPerTrain = 60 * len(branch) / d 
    trainRequired = np.ceil(fmin / servedPerTrain)
    for i in range(len(branch) - 1):
        flowGraph[branch[i]][branch[i + 1]] = max(0, flowGraph[branch[i]][branch[i + 1]] - trainRequired * servedPerTrain)
        flowGraph[branch[i + 1]][branch[i]] = max(0, flowGraph[branch[i]][branch[i + 1]] - trainRequired * servedPerTrain)
    if cyclic:
        flowGraph[branch[-1]][branch[0]] = max(0, flowGraph[branch[-1]][branch[0]] - trainRequired * servedPerTrain)
        flowGraph[branch[0]][branch[-1]] = max(0, flowGraph[branch[0]][branch[-1]] - trainRequired * servedPerTrain)
    return trainRequired, cyclic


def exhaustEdges(network, selectBranch):
    graph = initialGraph(network)
    influx = [[x[1] / station.spTime for x in station.spRate] for station in network.stations]
    statShapes = [station.shape for station in network.stations]
    flowGraph = buildFlowGraph(graph, statShapes, influx, network.shapes)
    trainNumber = 0

    while sum([sum(x) for x in flowGraph]) > 0.01:  #Float
        branch = selectBranch(flowGraph)
        (trainRequired, cyclic) = estimateTrainCost(branch, network, flowGraph)
        trainNumber += trainRequired
        trains = [Train(len(network.lines), 0, 0, [], 6)]
        line = Line(len(network.lines), branch, trains, cyclic)
        network.addLine(line)
    
    network.updateAllPaths()
    
    return trainNumber, len(network.lines)


def buildFlowRouteN(network, station, shapeNb, flowGraph, route):
    current = station.idt

    while len(route) != 0:
        (lineNb, dest) = route.pop()
        line = network.lines[lineNb]
        l = [i for i in range(len(line.route)) if line.route[i] == current]
        if len(l) == 0:
            print(line.route, current)
        k = min(l)
        count = 0

        while line.route[k] != dest and count < len(line.route):
            l = (k + 1) % len(line.route)
            s = network.stations[line.route[k]]
            t = network.stations[line.route[l]]
            flowGraph[s.idt][t.idt] += 60 * station.spRate[shapeNb][1] / station.spTime
            k = l
            count += 1
        
        current = dest


def buildFlowGraphN(network):
    n = len(network.stations)
    
    flowGraph = [[0 for _ in range(n)] for _ in range(n)]

    for station in network.stations:
        for i in range(len(station.paths)):
            buildFlowRouteN(network, station, i, flowGraph, station.paths[i][:])
    
    return flowGraph
