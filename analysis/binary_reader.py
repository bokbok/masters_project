import struct

DOUBLE_BYTES = 8
class Reader:

    def __init__(self, filename, meshsize):
        self.file = open(filename, 'rb')
        self.meshsize = meshsize
        self.work_out_frame_size()

    def read_points_dim(self, dim, points, tMax):
        data = {}
        for point in points:
            data[point] = []

        time = []
        index = self.dims.index(dim)

        offsets = self.determine_offsets(points, index)

        tCurr = 0

        while tCurr < tMax:
            bytes = self.file.read(DOUBLE_BYTES)
            if len(bytes) == 0:
                break
            t = struct.unpack('d', bytes)
            total_offset = 0
            for offset_with_point in offsets:
                total_offset += offset_with_point[0]
                self.file.seek(offset_with_point[0], 1)
                bytes = self.file.read(DOUBLE_BYTES)
                if len(bytes) == 0:
                    break
                val = struct.unpack('d', bytes)
                data[offset_with_point[1]].append(val[0])
                self.file.seek(-DOUBLE_BYTES, 1)

            self.file.seek(self.frame_size - total_offset, 1)

            tCurr = t[0]
            time.append(tCurr)
        return { 'time' : time, 'data' : data}

    def work_out_frame_size(self):
        header = self.file.readline()
        self.dims = header.strip().split(" ")
        self.frame_size = len(self.dims) * (self.meshsize ** 2) * DOUBLE_BYTES
        print self.dims

    def offsets(self, points, index):
        offsets = {}
        relative = []
        for point in points:
            offset = ((point[0] + point[1] * self.meshsize) * len(self.dims) + index) * DOUBLE_BYTES
            offsets[offset] = point
        return offsets



    def determine_offsets(self, points, index):
        relative = []
        offsets = self.offsets(points, index)
        sorted_offsets = sorted(offsets.keys())

        last = 0
        for offset in sorted_offsets:
            relative.append((offset - last, offsets[offset]))
            last = offset


        return relative

