import struct

DOUBLE_BYTES = 8
class Reader:

    def __init__(self, filename, dim, points, tMax, skip = 0, meshsize = None):
        self.file = open(filename, 'rb')
        self.meshsize = meshsize
        self.work_out_frame_size()
        self.dim = dim
        self.points = points
        self.tMax = tMax
        self.index = self.dims.index(dim)
        self.offsets = self.determine_offsets()
        self.skip = skip


    def readAll(self):
        data = {}
        time = []
        for point in self.points:
            data[point] = []

        self.forEach((lambda t, frame: self.remap(t, time, frame, data)))

        return { 'time' : time, 'data' : data}

    def forEach(self, callback):
        tCurr = 0

        while tCurr < self.tMax:
            tCurr, data = self.readNextFrame()
            if tCurr != None and data != None:
                callback(tCurr, data)
            else:
                break

    def readNextFrame(self):
        data = {}
        bytes = self.file.read(DOUBLE_BYTES)
        if len(bytes) == 0:
            t = None
        else:
            t = struct.unpack('d', bytes)
            t = t[0]
        total_offset = 0
        for offset_with_point in self.offsets:
            total_offset += offset_with_point[0]
            self.file.seek(offset_with_point[0], 1)
            bytes = self.file.read(DOUBLE_BYTES)
            if len(bytes) == 0:
                data = None
            else:
                val = struct.unpack('d', bytes)
                data[offset_with_point[1]]= val[0]
                self.file.seek(-DOUBLE_BYTES, 1)
        if t != None and data != None:
            self.file.seek((self.frame_size - total_offset) + self.skip * (self.frame_size + DOUBLE_BYTES), 1)

        return t, data


    def remap(self, t, time, frameData, data):
        time.append(t)
        for point in frameData.keys():
            data[point].append(frameData[point])


    def work_out_frame_size(self):
        header = self.file.readline()
        self.dims = header.strip().split(" ")
        if self.meshsize == None:
            print self.dims[-1]
            self.meshsize = int(self.dims[-1])
            self.dims.pop()
        self.frame_size = len(self.dims) * (self.meshsize ** 2) * DOUBLE_BYTES
        print self.dims

    def offsets(self):
        offsets = {}
        for point in self.points:
            offset = ((point[0] + point[1] * self.meshsize) * len(self.dims) + self.index) * DOUBLE_BYTES
            offsets[offset] = point
        return offsets



    def determine_offsets(self):
        relative = []
        offsets = self.offsets()
        sorted_offsets = sorted(offsets.keys())

        last = 0
        for offset in sorted_offsets:
            relative.append((offset - last, offsets[offset]))
            last = offset


        return relative

