from Glutton import *


def injections(n, N):
    if n == 0:
        return [[]]
    else:
        L = injections(n - 1, N)
        P = []
        for sigma in L:
            deja_vu = [False for _ in range(N)]
            for i in sigma:
                deja_vu[i] = True
            for i in range(N):
                if not deja_vu[i]:
                    P.append(sigma + [i])
        return P


def naiveTSP(network):
    for line in network.lines:
        n = len(line.route)
        L = injections(n,n)
        bestRoute = line.route
        bestWeight = totalWeight(network, bestRoute)
        for branch in L:
            w = totalWeight(network, bestRoute)
            if w < bestWeight:
                bestRoute = branch
                bestWeight = w
        line.route = bestRoute


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
    L = line.route

    for i in range(len(L)):
        bestRoute = L
        bestWeight = totalWeight(network, L)

        for j in range(i + 1, len(L)):

            newR = swap(L, i, j)
            newWeight = totalWeight(network, newR)

            if newWeight < bestWeight:
                bestRoute = newR
        
        L = bestRoute
    
    line.route = bestRoute


def optiOPT2(network):
    for line in network.lines:
        OPT2(network, line.nb)
    network.updateAllPaths()


def OPT3(network, lineNb):
    line = network.lines[lineNb]
    L = line.route

    for i in range(len(L)):
        bestRoute = L
        bestWeight = totalWeight(network, L)

        for j in range(i + 1, len(L)):

            for k in range(j + 1, len(L)):

                n1 = swap(L, i, j)
                n2 = swap(L, j, k)
                n3 = swap(L, i, k)
                n4 = swap(swap(swap(L, i, k), i, j), j, k)
                nL = [n1, n2, n3, n4]

                newWeight = min([totalWeight(network, newR) for newR in nL])
                i = min([i for i in range(4) if newWeight == totalWeight(network, nL[i])])

                if newWeight < bestWeight:
                    bestRoute = nL[i]
        
        L = bestRoute
    
    line.route = bestRoute


def optiOPT3(network):
    for line in network.lines:
        if line.cyclic:
            OPT3(network, line.nb)
    network.updateAllPaths()


def linesServing(network, station):
    L = [None for _ in network.lines]
    for line in network.lines:
        for i in range(len(line.route)):
            if line.route[i] == station.idt:
                L[line.nb] = i
    return L


def blunt(network, station):
    L = linesServing(network, station)
    n = sum([1 for x in L if x is not None])

    if n > 1:
        d = [0 for _ in network.lines]
        dist = network.distances

        for line in network.lines:
            if L[line.nb] is not None:
                i = L[line.nb]
                r = network.stations[line.route[(i - 1) % len(line.route)]]
                t = network.stations[line.route[(i + 1) % len(line.route)]]
                d[line.nb] = dist[r.idt][station.idt] + dist[station.idt][t.idt] - 2 * dist[r.idt][t.idt]
        
        M = max(d)

        if M != 0:
            j = max([k for k in range(len(network.lines)) if d[k] == M])
            line = network.lines[j]
            i = L[j]
            line.route = line.route[: i] + line.route[i + 1 :]
            return True

    return False


def optiBlunt(network):
    b = True

    while b:
        b = False
        for station in network.stations:
            b = b or blunt(network, station)
    
    network.updateAllPaths()


def test(p, f=3):
    n = randomEmptyNetwork(f, p)
    glutton(n, int(np.sqrt(p) - 1))
    plt.subplot(121)
    n.plot(False)
    optiOPT3(n)
    optiBlunt(n)
    optiOPT3(n)
    plt.subplot(122)
    n.plot()

