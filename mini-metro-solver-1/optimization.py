import numpy as np
import random as rnd
from sklearn.cluster import KMeans
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
        if goal is not None:
            station.durations[i] = - U.priority(goal)
        while goal is not None and goal % n != station.idt and len(route) < n:
            route.append((goal // n, goal % n))
            goal = spanningForest[goal]
        paths.append(route)

    return paths

class Passenger:
    def __init__(self, shape, shapeNb, route=[]):
        self.shape = shape
        self.shapeNb = shapeNb
        self.route = route
    
    def computeRoute(self, station):
        self.route = station.paths[self.shapeNb].copy()

class Station:
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

        for line_index, line in enumerate(self.lines):
            route = line.route if isinstance(line.route, list) else line.route.tolist()
            for k in range(len(route)):
                s = self.stations[route[k]]
                d = 0
                for l in range(1, len(route)):
                    t = self.stations[route[(k + l - 1) % len(route)]]
                    u = self.stations[route[(k + l) % len(route)]]
                    d += self.distances[t.idt][u.idt]
                    G[s.idt + n * line_index].append((u.idt + n * line_index, d))
        
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
    
    def addLine(self, line):
        self.lines.append(line)
        self.graph = self.createGraph()
    
    def addStation(self, station):
        self.stations.append(station)
        self.graph = self.createGraph()
    
    def addTrain(self, train):
        self.lines[train.line].trains.append(train)

class Line:
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

def calculate_optimal_routes(station_positions, station_shapes, number_of_lines):
    # 초기 설정
    n = len(station_positions)
    num_routes = number_of_lines

    # 거리 행렬 계산
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dmat[i, j] = np.sqrt(np.sum((np.array(station_positions[i]) - np.array(station_positions[j]))**2))
            else:
                dmat[i, j] = np.inf  # 자기 자신으로의 이동 시간은 무한대로 설정

    # k-means 클러스터링을 사용하여 중심점 찾기
    kmeans = KMeans(n_clusters=num_routes, random_state=42)
    kmeans.fit(station_positions)
    
    # 각 점에 가장 가까운 중심점에 할당
    assignments = kmeans.labels_

    # 각 경로에 할당된 점들 모으기
    cluster_points = [[] for _ in range(num_routes)]
    for i, cluster_idx in enumerate(assignments):
        cluster_points[cluster_idx].append(i)

    # 각 클러스터 내에서 최소 3가지 이상의 모양이 포함되도록 추가 연결
    for i in range(num_routes):
        unique_shapes = set()
        for idx in cluster_points[i]:
            unique_shapes.add(station_shapes[idx])
        if len(unique_shapes) < 3:  # 모양이 다른 점이 부족하면 추가 연결
            missing_shapes = list({'^', 's', 'o'} - unique_shapes)
            for shape in missing_shapes:
                closest_idx = None
                closest_dist = np.inf
                for j in range(n):
                    if j not in cluster_points[i] and station_shapes[j] == shape:
                        dist = np.min(dmat[j, cluster_points[i]])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_idx = j
                if closest_idx is not None:
                    cluster_points[i].append(closest_idx)

    # 초기 경로 설정
    Routes = [np.array(points) for points in cluster_points]

    # 메인 최적화 루프
    flag = True
    dold = 1e12
    iter = 0
    distHistory = []

    while flag:
        iter += 1
        totalDist = np.zeros(num_routes)
        
        for p in range(num_routes):
            route = Routes[p]
            d = dmat[route[-1], route[0]]
            for k in range(1, len(route)):
                d += dmat[route[k-1], route[k]]
            totalDist[p] = d
        
        minDist = np.sum(totalDist)
        distHistory.append(minDist)

        if iter % 200 == 0:
            if distHistory[-1] < dold:
                dold = distHistory[-1]
            else:
                flag = False

        # 최적의 경로 찾기 및 업데이트
        for p in range(num_routes):
            BestRoute = Routes[p]
            n_sub = len(BestRoute)
            randomIJ = np.sort(np.random.randint(0, n_sub, 2))
            I, J = randomIJ
            new_routes = [BestRoute.copy() for _ in range(4)]
            
            if I < J:
                new_routes[1][I:J+1] = BestRoute[I:J+1][::-1]  # 2-opt
            new_routes[2][[I, J]] = BestRoute[[J, I]]  # Swap
            if I+1 <= J:
                new_routes[3][I:J+1] = np.append(BestRoute[I+1:J+1], BestRoute[I])  # Shift
            
            # 3-opt 추가
            for _ in range(2):  # 3-opt 연산을 두 번 실행하여 변이 확률 증가
                randomIJK = np.sort(np.random.randint(0, n_sub, 3))
                i, j, k = randomIJK
                if i < j < k:
                    new_routes[3] = np.concatenate([BestRoute[:i+1], BestRoute[j:k+1], BestRoute[i+1:j], BestRoute[k+1:]])
            
            # 새로운 경로들 중 최적 경로 선택
            new_dists = np.zeros(4)
            for r in range(4):
                d = dmat[new_routes[r][-1], new_routes[r][0]]
                for k in range(1, n_sub):
                    d += dmat[new_routes[r][k-1], new_routes[r][k]]
                new_dists[r] = d
            
            best_new_route_idx = np.argmin(new_dists)
            Routes[p] = new_routes[best_new_route_idx]

    # 역 생성
    stations = []
    for i, (pos, shape) in enumerate(zip(station_positions, station_shapes)):
        spRate = [(shape, 1.0)]  # 각 역에 대한 출발 비율
        stations.append(Station(idt=i, shape=shape, waiting=[], lines=[], spRate=spRate, loc=pos))

    # Line 객체 생성
    line_objects = []
    for i, route in enumerate(Routes):
        line_objects.append(Line(nb=i, route=route.tolist(), trains=[]))

    # 네트워크 생성
    network = Network(stations=stations, distances=dmat, lines=line_objects, shapes=list(set(station_shapes)))

    # 각 역의 lines 속성 업데이트
    for line in line_objects:
        for station_id in line.route:
            if line.nb not in stations[station_id].lines:
                stations[station_id].lines.append(line.nb)

    # 네트워크의 그래프 재생성 및 경로 업데이트
    network.updateAllPaths()

    # 최적화된 노선 반환
    optimal_routes = []
    for route in Routes:
        optimal_routes.append([station_positions[i] for i in route])
    
    return optimal_routes, network
    
