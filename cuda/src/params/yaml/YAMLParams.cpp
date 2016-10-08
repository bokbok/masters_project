#include "YAMLParams.hpp"

//#include <yaml-cpp/parser.h>
//#include <yaml-cpp/iterator.h>
//#include <yaml-cpp/node.h>

#include <yaml-cpp/yaml.h>


#include <fstream>
#include <iostream>

using namespace YAML;

map<string, double> YAMLParams::read()
{
    ifstream fin(_filename.c_str());
    cout << "Open " << fin.is_open() << endl;
    Parser parser(fin);

    Node doc;

    map<string, double> result;

    while(parser.GetNextDocument(doc))
    {
    	const Node & params = doc["params"];
		for(Iterator it = params.begin(); it != params.end(); ++it)
		{
			string key;
    	    double val;
    	    it.first() >> key;
    	    it.second() >> val;
    	    result[key] = val;
    	    cout << key << " = " << val << endl;
    	}
    }

    return result;
}
