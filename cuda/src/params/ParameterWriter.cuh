/*
 * ParameterWriter.cuh
 *
 *  Created on: 12/07/2013
 *      Author: matt
 */

#ifndef PARAMETERWRITER_CUH_
#define PARAMETERWRITER_CUH_

#include <string>
#include "Params.cuh"

using namespace std;

class ParameterWriter
{
private:
	Params & _params;
	string _outputPath;

public:
	ParameterWriter(Params & params, string outputPath) :
		_params(params),
		_outputPath(outputPath)
	{

	}

	void write()
	{
		ofstream out;
		out.open((_outputPath + "/params.dat").c_str());

		ParameterSpace params = _params.params();
		StateSpace ics = _params.initialConditions();

		map<string, int> paramMap = _params.paramMap();
		map<string, int> stateMap = _params.stateMap();

		map<string, int>::iterator iter;

		out << "**** Parameters ****" << endl;
		for (iter = paramMap.begin(); iter != paramMap.end(); ++iter)
		{
			out << iter->first << " = " << params[iter->second] << endl;
		}

		out << endl << endl << endl;
		out << "**** ICs ****" << endl;
		for (iter = stateMap.begin(); iter != stateMap.end(); ++iter)
		{
			out << iter->first << " = " << ics[iter->second] << endl;
		}
	}

};


#endif /* PARAMETERWRITER_CUH_ */
