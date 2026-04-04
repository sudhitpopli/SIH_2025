import xml.etree.ElementTree as ET
import json
import os

def extract_roads(net_file, out_file):
    print(f"Parsing {net_file}...")
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    roads = []
    
    for edge in root.findall('edge'):
        # Skip internal edges if you want a cleaner map, but internal edges connect intersections.
        # Let's include all lanes for the full map view.
        for lane in edge.findall('lane'):
            shape_str = lane.get('shape')
            if shape_str:
                points = []
                for pt in shape_str.split(' '):
                    x, y = pt.split(',')
                    points.append([float(x), float(y)])
                roads.append(points)
    
    print(f"Extracted {len(roads)} road segments.")
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(roads, f)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    extract_roads(
        net_file="maps/connaught_place.net.xml",
        out_file="src/frontend/public/roads.json"
    )
