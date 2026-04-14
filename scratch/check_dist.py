import traci
import os
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_distances(net_file, radius=400):
    traci.start(['sumo', '-n', net_file])
    tls_ids = traci.trafficlight.getIDList()
    positions = {tls: traci.junction.getPosition(tls) for tls in tls_ids}
    
    adj_counts = []
    for t1 in tls_ids:
        neighbors = []
        for t2 in tls_ids:
            if t1 == t2: continue
            dist = calculate_distance(positions[t1], positions[t2])
            if dist <= radius:
                neighbors.append(t2)
        adj_counts.append(len(neighbors))
    
    avg = sum(adj_counts) / len(adj_counts)
    print(f"Distance-based (Radius={radius}m) Average Neighbors: {avg:.2f}")
    
    # Try 500m
    radius = 500
    adj_counts = []
    for t1 in tls_ids:
        neighbors = []
        for t2 in tls_ids:
            if t1 == t2: continue
            dist = calculate_distance(positions[t1], positions[t2])
            if dist <= radius:
                neighbors.append(t2)
        adj_counts.append(len(neighbors))
    avg = sum(adj_counts) / len(adj_counts)
    print(f"Distance-based (Radius={radius}m) Average Neighbors: {avg:.2f}")

    traci.close()

if __name__ == "__main__":
    check_distances('maps/connaught_place.net.xml')
