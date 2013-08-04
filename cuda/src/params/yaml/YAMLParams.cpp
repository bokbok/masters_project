#include "YAMLParams.hpp"
#include <yaml-cpp/parser.h>
#include <yaml-cpp/iterator.h>
#include <yaml-cpp/node.h>
#include <fstream>
#include <iostream>

map<string, double> YAMLParams::read()
{
    ifstream fin(_filename.c_str());
    cout << "Open " << fin.is_open() << endl;
    YAML::Parser parser(fin);

    YAML::Node doc;

    map<string, double> result;

    while(parser.GetNextDocument(doc))
    {
    	const YAML::Node & params = doc["params"];
		for(YAML::Iterator it = params.begin(); it != params.end(); ++it)
		{
			string key;
    	    double val;
    	    it.first() >> key;
    	    it.second() >> val;
    	    result[key] = val;
    	}
    }

    return result;
}
