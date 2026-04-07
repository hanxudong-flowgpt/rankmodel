#include <iostream>
#include <fstream>
#include <string>
#include "vector.pb.h"

int main() {
    std::ifstream fin( "../test.data" );
    //std::ifstream fin( "../testpb.data" );

    std::string s;
    while(getline(fin,s))
    {
        yyj::msg::VectorFeature vectorfea;
        vectorfea.ParseFromString(s);
        std::cout << "pk: " << vectorfea.pk() << std::endl;
        std::cout << "vec len: " << vectorfea.vec().size() << std::endl;
        auto vv = vectorfea.vec();
        for (int i = 0; i < vv.size(); i++) {
            std::cout << vv[i] << ",";
        }
        std::cout << std::endl;
    }
}