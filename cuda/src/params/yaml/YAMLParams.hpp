/*
 * YAMLParams.hpp
 *
 *  Created on: 04/08/2013
 *      Author: matt
 */

#ifndef YAMLPARAMS_HPP_
#define YAMLPARAMS_HPP_

#include <string>
#include <map>

using namespace std;

class YAMLParams
{
private:
	string _filename;

public:
	YAMLParams(string filename) :
		_filename(filename)
	{

	}

	map<string, double> read();

	string filename()
	{
		return _filename;
	}
};



#endif /* YAMLPARAMS_HPP_ */
