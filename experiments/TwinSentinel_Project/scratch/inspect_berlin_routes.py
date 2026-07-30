import gzip
with gzip.open("maps/berlin/berlin.rou.gz", "rt") as f:
    for i in range(50):
        line = f.readline()
        print(line, end="")
