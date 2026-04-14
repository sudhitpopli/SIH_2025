import traci
import os

def analyze_topology(net_file):
    traci.start(['sumo', '-n', net_file])
    tls_ids = set(traci.trafficlight.getIDList())
    print(f"Total Traffic Lights: {len(tls_ids)}")

    def get_downstream_neighbors(start_node, current_node, depth, max_depth, visited):
        if depth >= max_depth or current_node in visited:
            return set()
        visited.add(current_node)
        
        neighbors = set()
        try:
            out_edges = traci.junction.getOutgoingEdges(current_node)
            for edge in out_edges:
                if edge.startswith(":"): continue # skip internal
                target = traci.edge.getToJunction(edge)
                
                if target in tls_ids and target != start_node:
                    neighbors.add(target)
                else:
                    neighbors.update(get_downstream_neighbors(start_node, target, depth + 1, max_depth, visited))
        except Exception as e:
            pass
        return neighbors

    adj_counts = []
    print("Tracing Downstream...")
    for tls in list(tls_ids)[:10]: # Check first 10 for debug
        n = get_downstream_neighbors(tls, tls, 0, 10, {tls})
        print(f"  TLS {tls} -> {len(n)} neighbors: {n}")
        adj_counts.append(len(n))

    # Test full average
    full_adj = []
    for tls in tls_ids:
        n = get_downstream_neighbors(tls, tls, 0, 10, {tls})
        full_adj.append(len(n))
    
    avg = sum(full_adj) / len(full_adj)
    print(f"Average Downstream Neighbors (depth 10): {avg:.2f}")

    # Now test Bi-directional
    def get_upstream_neighbors(node, depth, max_depth, visited):
        if depth >= max_depth or node in visited:
            return set()
        visited.add(node)
        
        neighbors = set()
        try:
            in_edges = traci.junction.getIncomingEdges(node)
            for edge in in_edges:
                source = traci.edge.getFromJunction(edge)
                if source in tls_ids and source != tls:
                    neighbors.add(source)
                else:
                    neighbors.update(get_upstream_neighbors(source, depth + 1, max_depth, visited))
        except Exception:
            pass
        return neighbors

    bi_adj_counts = []
    for tls in tls_ids:
        d = get_downstream_neighbors(tls, 0, 10, {tls})
        u = get_upstream_neighbors(tls, 0, 10, {tls})
        bi_adj_counts.append(len(d | u))
    
    avg_bi = sum(bi_adj_counts) / len(bi_adj_counts)
    print(f"Average Bi-directional Neighbors (depth 10): {avg_bi:.2f}")

    traci.close()

if __name__ == "__main__":
    analyze_topology('maps/connaught_place.net.xml')
