import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from PriorityQueue import PriorityQueue



def computepaths(station, network):

    paths = []
    station.durations = [float('inf') for _ in network.shapes]
    for i in range(len(network.shapes)):
        if network.shapes[i] == station.shape:
            station.durations[i] = 0
        
    G = network.graph
    start = station.idt
    n = len(network.stations)
    p = len(network.lines)

    spanningForest = [None for _ in range(n * p)]
    dejaVu = [False for _ in range(n * p)]
    U = PriorityQueue(len(G))
    closer = [None for _ in network.shapes]

    for i in range(p):
        U.push(start + n * i, 0)
    
    while U.length() != 0 and len([x for x in closer if x is None]) != 0:
        
        (u, k) = U.pop()
        if dejaVu[u]:
            continue
        else:
            
            for i in range(len(network.shapes)):
                if closer[i] is None and network.stations[u % n].shape == network.shapes[i]:
                    closer[i] = u
            dejaVu[u] = True
            for (v, w) in G[u]:
                
                if - U.priority(v) > - k + w:
                    
                    U.changePrio(v, k - w)
                    spanningForest[v] = u
    
    for i in range(len(network.shapes)):
        route = []
        goal = closer[i]
        if not goal is None:
            station.durations[i] = - U.priority(goal)
        while goal is not None and goal % n != station.idt and len(route) < n:
            route.append((goal // n, goal % n))
            goal = spanningForest[goal]
        paths.append(route)

    return paths



class Passenger:

    '''A Passenger object represents an user of the metro network. It is described by the shape number of its destination and the route that it is planning to use. The route is given by a pile of tuples (line number, goal). '''

    def __init__(self, shape, shapeNb, route=[]):
        self.shape = shape
        self.shapeNb = shapeNb
        self.route = route
    
    def computeRoute(self, station):
        self.route = station.paths[self.shapeNb].copy()

        






class Station:

    '''A Station object represents a metro station. It is defined by its id, its shape, its capacity, its spawn rate of passengers, the lines that go through this station and the passengers currently waiting at the station. sp gives the different rates of spawning of shapes. The variable time increases 1 at a time to spTime and then returns to 0, and a random passenger is created then. If there are more passengers waiting than the capacity allowed, the variable overloadtime increases, and decreases to 0 otherwise.'''

    def __init__(self, idt, shape, waiting, lines, spRate, loc=None, spTime=100, capacity=8):
        self.idt = idt
        self.loc = loc
        self.shape = shape
        self.waiting = waiting
        self.time = 0
        self.spRate = spRate
        self.spTime = spTime
        self.capacity = capacity
        self.overloadTime = 0
        self.lines = lines
        self.transported = 0
        self.paths = []
        self.durations = []
    
    def updatePaths(self, network):
        self.paths = computepaths(self, network)
    
    def upCrowded(self, network):
        if self.overloadTime >= 100:
            network.end = True
        elif self.capacity < len(self.waiting):
            self.overloadTime += 1
        elif self.overloadTime > 0:
            self.overloadTime -= 1
    
    def count(self):
        self.transported += 1
    
    def spawn(self, network):
        if self.time == self.spTime:
            r = rnd.random()
            for i in range(len(self.spRate)):
                (shape, ratio) = self.spRate[i]
                if r < ratio:
                    passenger = Passenger(shape, i)
                    passenger.computeRoute(self)
                    self.waiting.append(passenger)
                    break
                elif r >= ratio:
                    r -= ratio
            self.time = 0
        else:
            self.time += 1



class Network:

    '''A Map object is a graph representing a metro network. It is given by the list of its vertex, that are the stations, an array of the times it costs to travel between any pair of stations (integer) and the metro lines currently working.'''

    def __init__(self, stations, distances, lines, shapes):
        self.shapes = shapes
        self.stations = stations
        self.distances = distances
        self.lines = lines
        self.end = False
        self.graph = self.createGraph()

    def nextState(self):
        for station in self.stations:
            station.spawn(self)
            station.upCrowded(self)
        for line in self.lines:
            line.nextState(self.distances, self.stations)
    
    def oneEternityLater(self, n):
        while n > 0 and not self.end:
            self.nextState()
            n -= 1
        
        return self.end
    
    def createGraph(self):
        G = [[] for _ in self.lines for _ in self.stations]
        n = len(self.stations)

        for line in self.lines:
            for k in range(len(line.route)):
                s = self.stations[line.route[k]]
                d = 0
                for l in range(1, len(line.route)):
                    t = self.stations[line.route[(k + l - 1) % len(line.route)]]
                    u = self.stations[line.route[(k + l) % len(line.route)]]
                    d += self.distances[t.idt][u.idt]
                    G[s.idt + n * line.nb].append((u.idt + n * line.nb, d))
        
        for s in self.stations:
            for line1 in s.lines:
                for line2 in s.lines:
                    if line1 != line2:
                        G[s.idt + n * line1].append((s.idt + n * line2, self.lines[line2].waitingTime(self)))

        return G
    
    def updateAllPaths(self):
        for station in self.stations:
            station.lines = [line.nb for line in self.lines if station.idt in line.route]
        self.graph = self.createGraph()
        for station in self.stations:
            station.updatePaths(self)
    
    def plot(self, show=True):

        shapeList = ["s", "^", "o", "p", "P", "*", "d"]

        for line in self.lines:
            X = [self.stations[i].loc[0] for i in line.route]
            Y = [self.stations[i].loc[1] for i in line.route]
            X.append(X[0])
            Y.append(Y[0])
            plt.plot(X, Y, linewidth=2)
        
        for station in self.stations:
            (x,y) = station.loc
            plt.scatter(x, y, s=64, c='black', marker=shapeList[station.shape])
        
        plt.axis('equal')
        if show:
            plt.show()
    
    def addLine(self, line):
        self.lines.append(line)
        self.graph = self.createGraph()
    
    def addStation(self, station):
        self.stations.append(station)
        self.graph = self.createGraph()
    
    def addTrain(self, train):
        self.lines[train.line].trains.append(train)







class Line:

    '''A Line object represents a metro line. It is defined by an unique number, the list of stations on this line (which can be cyclic) and the list of trains on this line. The direct parameter indicates if the train is following the route in the left -> right order or in the opposite order.'''

    def __init__(self, nb, route, trains, cyclic=True):
        self.nb = nb
        self.route = route
        self.trains = trains
        self.cyclic = cyclic
    
    def nextState(self, dist, stations):
        for train in self.trains:
            if train.nextTime != 0:
                train.nextTime -= 1
            else:
                station = stations[self.route[train.nextDest]]
                if not self.cyclic and train.nextDest == len(self.route) - 1:
                    train.direct = False
                elif not self.cyclic and train.nextDest == 0:
                    train.direct = True
                if train.direct:
                    train.nextDest = (train.nextDest + 1) % len(self.route)
                else:
                    train.nextDest = (train.nextDest - 1) % len(self.route)
                nextStation = stations[self.route[train.nextDest]]
                train.nextTime = dist[station.idt][nextStation.idt]
                train.empty(station)
                train.fill(station)
    
    def waitingTime(self, network):
        return sum([network.distances[self.route[k]][self.route[(k + 1) % len(self.route)]] for k in range(len(self.route))]) / (1 + 2 * len(self.trains))





class Train:

    '''A Train object represents a metro train. It is defined by the number of the line it is working on, its next destination, the time needed to go to the next destination, the list of passengers onboard and its capacity.'''

    def __init__(self, line, nextDest, nextTime, passengers, capacity, direct=True):
        self.line = line
        self.nextDest = nextDest
        self.nextTime = nextTime
        self.passengers = passengers
        self.capacity = capacity
        self.direct = direct
    
    def empty(self, station):
        i = 0
        stillGoing = []
        while i < len(self.passengers):
            passenger = self.passengers[i]
            if passenger.shape == station.shape:
                station.count()
            elif passenger.route[-1][1] == station.idt:
                passenger.route.pop()
                station.waiting.append(passenger)
            else:
                stillGoing.append(passenger)
            i += 1
        self.passengers = stillGoing
    
    def fill(self, station):
        i = 0
        stillWaiting = []
        while i < len(station.waiting) and self.capacity > len(self.passengers):
            passenger = station.waiting[i]
            if passenger.route[-1][0] == self.line:
                self.passengers.append(passenger)
            else:
                stillWaiting.append(passenger)
            i += 1
        station.waiting = stillWaiting


    