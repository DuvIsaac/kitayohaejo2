from Structures import *


def randomLoc(radius=50):
    x = radius * rnd.random()
    y = radius * rnd.random()
    return (x, y)


def randomRate(nbShapes, shape):
    rates = [rnd.random() for _ in range(shape)] + [0] + [rnd.random() for _ in range(nbShapes - shape - 1)]
    s = sum(rates)
    return [(i, rates[i] / s) for i in range(len(rates))]


def randomStation(idt, loc, nbShapes, spTimeRange, capacityRange, spRate=None):
    shape = rnd.randint(0, 2)
    spTime = rnd.randint(spTimeRange[0], spTimeRange[1])
    capacity = rnd.randint(capacityRange[0], capacityRange[1])
    if spRate is None:
        spRate = randomRate(nbShapes, shape)
    return Station(idt, shape, [], [], spRate, loc, spTime, capacity)


def buildDistances(locations, dist):
    return [[np.ceil(dist(x,y)) for y in locations] for x in locations]


def euclideanDist(a, b):
    (x, y) = a
    (u, v) = b
    return np.sqrt((x - u) ** 2 + (y - v) ** 2)


def randomEmptyNetwork(nbShapes, nbStations, locations=None, spTimeRanges=None, capacityRanges=None, spRates=None, dist=euclideanDist):
    stations = []
    if locations is None:
        locations = [randomLoc() for _ in range(nbStations)]
    if spTimeRanges is None:
        spTimeRanges = [[25, 30] for _ in range(nbStations)]
    if capacityRanges is None:
        capacityRanges = [[8, 8] for _ in range(nbStations)]
    if spRates is None:
        spRates = [None for _ in range(nbStations)]

    for i in range(nbStations):
        station = randomStation(i, locations[i], nbShapes, spTimeRanges[i], capacityRanges[i], spRates[i])
        stations.append(station)
    distances = buildDistances(locations, dist)

    for i in range(3, nbShapes):
        stations[i - 3].shape = i
        stations[i - 3].spRate = randomRate(nbShapes, i)

    return Network(stations, distances, [], [k for k in range(nbShapes)])
