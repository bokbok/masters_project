import matplotlib.pyplot as plt
import re
import sys


file = open(sys.argv[1])
yAxis = sys.argv[2]
tMax = float(sys.argv[3])

points = []
t = []

data = {}
pointNum = 4

while len(sys.argv) > pointNum:
    point = sys.argv[pointNum].split(',')
    points.append(point)
    data[sys.argv[pointNum]] = []
    pointNum += 1


dataMin = None
dataMax = None
index = None


for line in file:
    if index == None:
        headers = line.split(' ')
        index = headers.index(yAxis)

    else:
        tMatch = re.match( r'^t=(.*)$', line)
        if tMatch:
            tVal = float(tMatch.group(1))
            if tVal > tMax:
                break
            t.append(float(tMatch.group(1)))
            print "t = " + tMatch.group(1)
        pointNum = 0
        for coord in points:
            if line.startswith('(' + coord[0] + ',' + coord[1] + ')'):
                pointRegex = r'^\(' + coord[0] + ',' + coord[1] + r'\):(.*)$'
                pointMatch = re.match(pointRegex, line)
                dataLine = pointMatch.group(1)
                elements = dataLine.split(' ')
                element = float(elements[index])
                data[coord[0] + "," + coord[1]].append(element)
                if dataMax == None or element > dataMax:
                    dataMax = element
                if dataMin == None or element < dataMin:
                    dataMin = element
                pointNum += 1



for point in data.keys():
    plt.plot(t, data[point], label = point)

plt.legend()
plt.ylabel(yAxis)
plt.xlabel('t')
plt.axis([t[0], t[-1], dataMin, dataMax])
plt.show()