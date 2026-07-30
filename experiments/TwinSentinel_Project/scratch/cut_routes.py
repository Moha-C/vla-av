import gzip
import os

def cut_route_file(src_path, dest_path, max_depart=360.0):
    print(f"Filtering {src_path} -> {dest_path} (max_depart={max_depart}s)...")
    
    is_gzip = src_path.endswith(".gz")
    open_func = gzip.open if is_gzip else open
    mode = "rt" if is_gzip else "r"
    
    with open_func(src_path, mode) as infile, open(dest_path, "w") as outfile:
        in_vehicle = False
        skip_vehicle = False
        veh_kept = 0
        veh_skipped = 0
        line_count = 0
        
        for line in infile:
            line_count += 1
            stripped = line.strip()
            
            if stripped.startswith("<vehicle") or stripped.startswith("<flow"):
                in_vehicle = True
                depart_val = 0.0
                try:
                    # Parse depart or begin time
                    attr = 'depart="' if 'depart="' in stripped else 'begin="'
                    if attr in stripped:
                        parts = stripped.split(attr)
                        dep_str = parts[1].split('"')[0]
                        depart_val = float(dep_str)
                except Exception as e:
                    print(f"Error parsing depart on line {line_count}: {e}")
                    
                if depart_val > max_depart:
                    skip_vehicle = True
                    veh_skipped += 1
                else:
                    skip_vehicle = False
                    veh_kept += 1
                    
                if not skip_vehicle:
                    outfile.write(line)
            elif stripped.startswith("</vehicle>") or stripped.startswith("</flow>"):
                if not skip_vehicle:
                    outfile.write(line)
                in_vehicle = False
                skip_vehicle = False
            else:
                if in_vehicle:
                    if not skip_vehicle:
                        outfile.write(line)
                else:
                    outfile.write(line)
                    
    print(f"  Done. Kept {veh_kept}, skipped {veh_skipped}.")

def main():
    # 1. Cut berlin.rou.gz -> berlin_cut.rou.xml
    cut_route_file("maps/berlin/berlin.rou.gz", "maps/berlin/berlin_cut.rou.xml")
    
    # 2. Cut berlin_bus.rou.xml -> berlin_bus_cut.rou.xml
    cut_route_file("maps/berlin/berlin_bus.rou.xml", "maps/berlin/berlin_bus_cut.rou.xml")

if __name__ == "__main__":
    main()
