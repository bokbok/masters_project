import matplotlib.pyplot as plt
import re
import sys


def regex(x, y):
    return r'^\(' + str(x) + ',' + str(y) + r'\):(.*)$'

file = open(sys.argv[1])
x = int(sys.argv[2])
y = int(sys.argv[3])
yAxis = sys.argv[4]

tMax = None

if len(sys.argv) > 5:
    tMax = float(sys.argv[5])

t = []
data = []
avg = []

dataMin = None
dataMax = None
index = None

vals = []
pointVal = None
deltaX = 0.1

for line in file:
    if index == None:
        headers = line.split(' ')
        index = headers.index(yAxis)

    else:
        tMatch = re.match( r'^t=(.*)$', line)

        for off in range(-2, 3):
            for regexp in [regex(x + off, y), regex(x, y + off)]:
                match = re.match(regexp, line)
                if match:
                    dataLine = match.group(1)
                    val = float(dataLine[index])
                    if off != 0:
                        mult = (16 if off == 1 or off == -1 else -1)
                        vals.append(val * mult)
                    else:
                        pointVal = val

        if tMatch:
            tVal = float(tMatch.group(1))
            if tVal > tMax:
                break

            if len(vals) == 8:
                laplacian = (sum(vals) - 60 * pointVal) / (12 * (deltaX ** 2))
                data.append(laplacian)
                pointVal = None
                vals = []
                t.append(tVal)

                if len(data) > 50:
                    sumV = 0
                    for i in range(50):
                        sumV += data[-(i + 1)]
                    avg.append(sumV / 50)
                else:
                    avg.append(laplacian)

                if dataMax == None or laplacian > dataMax:
                    dataMax = laplacian
                if dataMin == None or laplacian < dataMin:
                    dataMin = laplacian

            print "t = " + tMatch.group(1)


print len(t), len(data)
plt.plot(t, data)
plt.plot(t, avg)
plt.ylabel(yAxis)
plt.xlabel('t')
plt.axis([t[0], t[-1], dataMin, dataMax])
plt.show()