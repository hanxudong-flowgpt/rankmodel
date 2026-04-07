proto_library(
        name = "vector_proto",
        srcs = [
                "vector.proto",
        ],
)

cc_proto_library(
        name = "vector_cc_proto",
        deps = [":vector_proto"],
)
cc_binary(
    name = "test_decode",
    srcs = ["test_decode.cpp"],
    deps = [
        ":vector_cc_proto"
    ],
)