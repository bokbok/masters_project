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

template <class T>
class ParameterWriter
{
private:
	ParameterMesh<T> * _parameterMesh;
	Params<T> * _params;
	string _outputPath;
	double _t, _deltaX, _deltaT, _effectiveDeltaT, _randomiseFraction;
	int _meshSize;

public:
	ParameterWriter(double t,
					int meshSize,
					double deltaT,
					double effectiveDeltaT,
					double deltaX,
					double randomiseFraction,
					Params<T> * params,
					ParameterMesh<T> * parameterMesh,
					string outputPath) :
		_t(t),
		_meshSize(meshSize),
		_parameterMesh(parameterMesh),
		_params(params),
		_outputPath(outputPath),
		_deltaX(deltaX),
		_deltaT(deltaT),
		_effectiveDeltaT(effectiveDeltaT),
		_randomiseFraction(randomiseFraction)
	{
	}

	void write()
	{
		ofstream out;
		out.open((_outputPath + "/run.info").c_str());

		ParameterSpace params = _parameterMesh->paramsAt(0, 0);
		StateSpace ics = _params->initialConditions();

		map<string, int> paramMap = _params->paramMap();
		map<string, int> stateMap = _params->stateMap();

		out << "**** Integration Params ****" << endl;
		out << "Runge-Kutta fourth order" << endl;
		out << "simulation length = " << _t << endl;
		out << "simulation delta_t = " << _deltaT << endl;
		out << "output file delta_t = " << _effectiveDeltaT << endl;
		out << "spatial spacing = " << _deltaX << endl;
		out << "mesh size = " << _meshSize << endl;
		out << "randomisation = " << _randomiseFraction << endl;
		out << endl << endl << endl;

		map<string, int>::iterator iter;

		out << "**** Parameters ****" << endl;
		out << "Details: " << _params->describe() << endl;
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
