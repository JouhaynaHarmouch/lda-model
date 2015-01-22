//============================================================================
// Name        : M2M3.cpp
// Type        : main
// Created by  : Furong Huang  and Forough Arabshahi
// Version     :
// Copyright   : Copyright (c) 2013 Furong Huang and Forough Arabshahi.
//               All rights reserved
// Description : Single Node ALS Topic Modeling
//============================================================================

#include "stdafx.h"
#include <stdlib.h>
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;
clock_t TIME_start, TIME_end;
int NX;
int NA;
int KHID;
double alpha0;
int DATATYPE;
int main(int argc, const char * argv[])
{

	NA = furong_atoi(argv[1]);
	KHID = furong_atoi(argv[2]);
	DATATYPE = furong_atoi(argv[3]);
	//===============================================================================================================================================================
	// User Manual:
	// (1) Data specs
	// NA is the vocabulary size
	// KHID is the number of topics you want to learn
	// DATATYPE denotes the index convention.
	// -> DATATYPE == 1 assumes MATLAB index which starts from 1,DATATYPE ==0 assumes C++ index which starts from 0 .
	// e.g. 500 3 1

	// (2) Input files
	// FILE_M3 denotes the filename for the third order moment
	// $(SolutionDir)\datasets\$(CorpusName)\M3.txt
	// e.g. $(SolutionDir)datasets\synthetic\M3.txt $(SolutionDir)datasets\synthetic\M2.txt

	const char* FILE_eigVal_WRITE = argv[4];
	const char* FILE_eigVect_WRITE = argv[5];
	// (3) Output files
	// FILE_eigVal_WRITE denotes the filename for eigVal
	// FILE_Vect_WRITE denotes the filename for eigVect

	const char* FILE_M3 = argv[6];

	// The format is:
	// $(SolutionDir)\datasets\$(CorpusName)\result\eigVal.txt
	// $(SolutionDir)\datasets\$(CorpusName)\result\eigVect.txt

	// e.g. $(SolutionDir)datasets\synthetic\result\eigVal.txt $(SolutionDir)datasets\synthetic\result\eigVect.txt

	//==============================================================================================================================================================

	cout << "(1) Reading data----------------" << endl;
	TIME_start = clock();

	MatrixXd input_M3(NA, NA * NA);
	input_M3 = MatrixXd::Zero(NA, NA*NA);

	input_M3 = read_G_sparse((char*) FILE_M3, "third order moment", NA, NA * NA);
    cout << "input_tensor: " << endl << input_M3 << endl;
	TIME_end = clock();
	double time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices before preproc = %5.10e (Seconds)\n", time_readfile);

	MatrixXd T(NA, NA * NA);
	T = MatrixXd(input_M3);

	cout << "(2) Tensor decomposition----------------" << endl;
	TIME_start = clock();
	VectorXd eigVal(KHID);
	MatrixXd eigVect(KHID, KHID);

    bool fail=1;
    int restart_num = 0;
    while(fail and restart_num<10){
        cout << "Running ALS " << restart_num << endl;
        fail = tensorDecom_batchALS(T, eigVal, eigVect);
        restart_num +=1;
    }
    

	cout << "K space eigenvectors: " << endl << eigVect << endl;
	cout << "K space eigenvalues: " << endl << eigVal << endl;

	cout << "(3) Writing results----------" << endl;
	write_alpha((char *)FILE_eigVal_WRITE, eigVal);
	write_beta((char *)FILE_eigVect_WRITE, eigVect);


	cout << "(4) Program over------------" << endl;
	return 0;
}
