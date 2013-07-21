import matplotlib.pyplot as plt
import re
import sys


file = open(sys.argv[1])
x = int(sys.argv[2])
y = int(sys.argv[3])
yAxis = sys.argv[4]

tMax = None

if len(sys.argv) > 5:
    tMax = float(sys.argv[5])

pointRegex = r'^\(' + str(x) + ',' + str(y) + r'\):(.*)$'
northRegex = r'^\(' + str(x) + ',' + str(y + 1) + r'\):(.*)$'
southRegex = r'^\(' + str(x) + ',' + str(y - 1) + r'\):(.*)$'
eastRegex = r'^\(' + str(x + 1) + ',' + str(y) + r'\):(.*)$'
westRegex = r'^\(' + str(x - 1) + ',' + str(y) + r'\):(.*)$'

t = []
data = []
avg = []

dataMin = None
dataMax = None
index = None
pointVal = None
northVal = None
southVal = None
eastVal = None
westVal = None

deltaX = 0.1

print pointRegex

for line in file:
    if index == None:
        headers = line.split(' ')
        index = headers.index(yAxis)

    else:
        tMatch = re.match( r'^t=(.*)$', line)
        pointMatch = re.match(pointRegex, line)
        northMatch = re.match(northRegex, line)
        southMatch = re.match(southRegex, line)
        eastMatch = re.match(eastRegex, line)
        westMatch = re.match(westRegex, line)
        
        if tMatch:
            tVal = float(tMatch.group(1))
            if tVal > tMax:
                break


            if northVal and southVal and eastVal and westVal:
                laplacian = (northVal + southVal + eastVal + westVal - 4 * pointVal) / (deltaX ** 2)
                data.append(laplacian)
                pointVal = None
                northVal = None
                southVal = None
                eastVal = None
                westVal = None
                t.append(tVal)

                if len(data) > 50:
                    sum = 0
                    for i in range(50):
                        sum += data[-(i + 1)]
                    avg.append(sum / 50)
                else:
                    avg.append(laplacian)

                if dataMax == None or laplacian > dataMax:
                    dataMax = laplacian
                if dataMin == None or laplacian < dataMin:
                    dataMin = laplacian

            print "t = " + tMatch.group(1)
        if northMatch:
            dataLine = northMatch.group(1)
            northVal = float(dataLine[index])
        if southMatch:
            dataLine = southMatch.group(1)
            southVal = float(dataLine[index])
        if eastMatch:
            dataLine = eastMatch.group(1)
            eastVal = float(dataLine[index])
        if westMatch:
            dataLine = westMatch.group(1)
            westVal = float(dataLine[index])
        if pointMatch:
            dataLine = pointMatch.group(1)
            pointVal = float(dataLine[index])
            


print len(t), len(data)
plt.plot(t, data)
plt.plot(t, avg)
plt.ylabel(yAxis)
plt.xlabel('t')
plt.axis([t[0], t[-1], dataMin, dataMax])
plt.show()