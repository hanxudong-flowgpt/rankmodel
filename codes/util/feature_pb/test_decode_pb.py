from vector_pb2 import VectorFeature
import base64

for line in open('testpb.data'):
    a = VectorFeature()
    #a.ParseFromString(base64.decodestring(line))
    a.ParseFromString(line[0:-1])
    print a.pk
    print len(a.vec)
    assert len(a.vec) == 128
    # print(serializeToString, type(serializeToString))