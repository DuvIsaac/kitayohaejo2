import numpy as np
import random as rnd
from Structures import Station, Network
from NetworkBuilder import randomEmptyNetwork, buildDistances, euclideanDist
from Glutton import glutton, totalWeight
from OPT import optiOPT3, optiBlunt

def ensure_all_stations_connected(network):
    all_stations = set(station.idt for station in network.stations)
    connected_stations = set()
    
    for line in network.lines:
        connected_stations.update(line.route)
    
    unconnected_stations = all_stations - connected_stations
    
    for station_id in unconnected_stations:
        closest_line = None
        closest_distance = float('inf')
        for line in network.lines:
            for connected_station_id in line.route:
                distance = network.distances[station_id][connected_station_id]
                if distance < closest_distance:
                    closest_distance = distance
                    closest_line = line
        
        if closest_line:
            insertion_index = np.argmin([network.distances[station_id][closest_station_id] for closest_station_id in closest_line.route])
            closest_line.route.insert(insertion_index, station_id)
            network.updateAllPaths()

def ensure_diverse_shapes_in_lines(network):
    for line in network.lines:
        shapes_in_line = set(network.stations[station_id].shape for station_id in line.route)
        for shape in network.shapes:
            if shape not in shapes_in_line:
                for station in network.stations:
                    if station.shape == shape and station.idt not in line.route:
                        insertion_index = np.argmin([network.distances[station.idt][line_station_id] for line_station_id in line.route])
                        line.route.insert(insertion_index, station.idt)
                        shapes_in_line.add(shape)
                        network.updateAllPaths()
                        break

def add_missing_connections(network):
    for station in network.stations:
        connected_shapes = set(network.stations[neighbor_id].shape for neighbor_id in station.lines if station.idt != neighbor_id)
        missing_shapes = set(network.shapes) - connected_shapes
        
        for shape in missing_shapes:
            closest_station = None
            closest_distance = float('inf')
            for other_station in network.stations:
                if other_station.shape == shape and other_station.idt not in station.lines:
                    distance = network.distances[station.idt][other_station.idt]
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_station = other_station
            
            if closest_station:
                closest_line = None
                for line in network.lines:
                    if station.idt in line.route or closest_station.idt in line.route:
                        closest_line = line
                        break
                
                if closest_line:
                    if station.idt not in closest_line.route:
                        closest_line.route.append(station.idt)
                    if closest_station.idt not in closest_line.route:
                        closest_line.route.append(closest_station.idt)
                    network.updateAllPaths()

def ensure_minimum_connections_per_station(network):
    for station in network.stations:
        if len(station.lines) == 0:
            # 연결된 노선이 없는 경우, 다른 노선에 추가
            closest_line = None
            closest_distance = float('inf')
            for line in network.lines:
                for connected_station_id in line.route:
                    distance = network.distances[station.idt][connected_station_id]
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_line = line
            
            if closest_line:
                insertion_index = np.argmin([network.distances[station.idt][line_station_id] for line_station_id in closest_line.route])
                closest_line.route.insert(insertion_index, station.idt)
                network.updateAllPaths()
        
        # 각 역이 최소한 두 개의 다른 모양의 역과 연결되도록 보장
        connected_shapes = set(network.stations[neighbor_id].shape for neighbor_id in station.lines if station.idt != neighbor_id)
        missing_shapes = set(network.shapes) - connected_shapes
        for shape in missing_shapes:
            closest_station = None
            closest_distance = float('inf')
            for other_station in network.stations:
                if other_station.shape == shape and other_station.idt not in station.lines:
                    distance = network.distances[station.idt][other_station.idt]
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_station = other_station
            
            if closest_station:
                closest_line = None
                for line in network.lines:
                    if station.idt in line.route or closest_station.idt in line.route:
                        closest_line = line
                        break
                
                if closest_line:
                    if station.idt not in closest_line.route:
                        closest_line.route.append(station.idt)
                    if closest_station.idt not in closest_line.route:
                        closest_line.route.append(closest_station.idt)
                    network.updateAllPaths()

def two_opt(route, network):
    best_route = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if totalWeight(network, new_route) < totalWeight(network, best_route):
                    best_route = new_route
                    improved = True
        route = best_route
    return best_route

def further_optimize(network):
    for line in network.lines:
        initial_route = line.route.copy()
        best_route = two_opt(initial_route, network)
        line.route = best_route
        network.updateAllPaths()

def calculate_optimal_routes(station_positions, station_shapes, number_of_lines):
    # 역 및 노선 초기화
    nbShapes = len(set(station_shapes))
    nbStations = len(station_positions)
    
    # 역 생성
    stations = []
    for i, (pos, shape) in enumerate(zip(station_positions, station_shapes)):
        spRate = [(shape, 1.0)]  # 각 역에 대한 출발 비율
        stations.append(Station(idt=i, shape=shape, waiting=[], lines=[], spRate=spRate, loc=pos))

    # 거리 계산
    distances = buildDistances(station_positions, euclideanDist)

    # 네트워크 생성
    network = Network(stations=stations, distances=distances, lines=[], shapes=list(set(station_shapes)))

    # 초기 노선 생성
    glutton(network, number_of_lines)

    # 연결되지 않은 역을 연결
    ensure_all_stations_connected(network)

    # 각 노선 내에 다양한 모양의 역들이 포함되도록 조정
    ensure_diverse_shapes_in_lines(network)

    # 누락된 연결 추가
    add_missing_connections(network)

    # 각 역이 최소한 하나의 노선과 연결되도록 보장
    ensure_minimum_connections_per_station(network)

    # 최적화
    optiOPT3(network)
    optiBlunt(network)
    optiOPT3(network)

    # 추가 최적화
    further_optimize(network)
    
    # 최적화된 노선 반환
    optimal_routes = []
    for line in network.lines:
        optimal_routes.append([network.stations[station_id].loc for station_id in line.route])
    
    return optimal_routes, network
