#define SAMPLES_N 30
#define LLS_DEBUG
#include <iostream>
#include "../PolyLLS.hpp"

using namespace Eigen;
using namespace std;


PolyLLS lls(3);
IOFormat fmt(4, 0, " ", "\n", "", "", "[", "]");

MatrixXd testInput(SAMPLES_N, 2);
VectorXd testOutput(SAMPLES_N);

double testFunction(double x1, double x2){
    return 2.0 + 4*x1 + pow(x1,2) + x2 + -pow(x2,3);
}

int main(int argc, char const *argv[])
{
    for (size_t i = 0; i < SAMPLES_N; i++)
    {
        double x1 = rand()%10000;
        double x2 = rand()%10000;
        testInput(i,0) = x1;
        testInput(i,1) = x2;
        testOutput(i) = testFunction(x1, x2);
    }
    cout << "Inputs: " << endl;
    cout << testInput.format(fmt) << endl;
    cout << "Outputs: " << endl;
    cout << testOutput.format(fmt) << endl;

    lls.setData(testInput, testOutput);
    return 0;
}

