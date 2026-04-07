from vector_pb2 import VectorFeature
import numpy as np

a = np.ones((10, 512))
b = a.tolist()
c = []
for i in range(len(b)):
    d = VectorFeature()
    d.pk = i
    for v in b[i]:
        d.vec.append(v)
    c.append(d)

with open('test.data', 'w') as f:
    for x in c:
        serializeToString = x.SerializeToString()
        f.write(serializeToString + '\n')
        #f.write(serializeToString)
        # print(serializeToString, type(serializeToString))