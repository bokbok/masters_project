from binary_reader import Reader, HeaderInfo
from renderer import Renderer
import sys
import ntpath


filename = sys.argv[1]
dim = sys.argv[2]
tMax = float(sys.argv[3])
step = int(sys.argv[4])
skip = int(sys.argv[5])

points = []

header = HeaderInfo(filename)

for x in range(0, header.meshsize, step):
    for y in range(0, header.meshsize, step):
        points.append((x, y))

reader = Reader(filename, dim, points, tMax, skip = skip)

renderer = Renderer(reader, ntpath.dirname(filename) + '/render.mp4', step, tMax)

renderer.render()
