#include <eigen3/Eigen/Dense>

using namespace Eigen;

class PolyLLS
{
private:
    MatrixXd x;     // Samples
    VectorXd y;     // Output
    MatrixXd phi;   // Function Terms
    VectorXd theta; // Ideal Params
    int order;
    int M; // # of params
    int N; // # of samples
    void computeParams();

public:
    IOFormat fmt;
    PolyLLS(int order);
    void setData(MatrixXd input, VectorXd output);
    double computeMapped();
};

PolyLLS::PolyLLS(int order)
{
    this->fmt = IOFormat(4, 0, " ", "\n", "", "", "[", "]");
    this->order = order;
}

void PolyLLS::setData(MatrixXd input, VectorXd output)
{
    // Update # of params
    this->M = this->order*input.cols() +1;

    // Fill out y
    this->y.resize(output.rows());
    y = output;

    // Fill out phi with basis function terms
    this->N = input.rows();
    this->phi.resize(this->N, this->M);
    /**
     * Phi will be the polynomial entries of each basis function concatenated horizontally, example below for a 3rd order polynomial with 2 dimensions
     * i.e. the basis function: F(x1,x2) = a0 + a1*x1 + a2*x1^2 + a3*x1^3 + b1*x2 + b2*x2^2 + b3*x2^3
     * with a0,a1.... being coefficients for input x1 & b0,b1.... being coefficients for input x2
     *
     * [ 1 x1_1 x1_1^2 x1_1^3 | x2_1 x2_1^2 x2_1^3 ]
     * [ 1 x1_2 x1_2^2 x1_2^3 | x2_2 x2_2^2 x2_2^3 ]
     * [ 1 x1_3 x1_3^2 x1_3^3 | x2_3 x2_3^2 x2_3^3 ]
     */
    for (int col = 0; col < M; col++)
    {
        for (int row = 0; row < N; row++)
        {
            double entry;
            if (col == 0)
                entry = 1;
            else
            {
                int localIndex = ((col - 1) % (order));     // Index within dimension partition
                int dimensionIndex = ((col - 1) / (order)); // Dimension partition of phi
                if (localIndex == 0)
                    entry = input(row, dimensionIndex);
                else
                    entry = pow(input(row, dimensionIndex), localIndex + 1);
            }
            this->phi(row, col) = entry;
        }
    }
#if defined(_GLIBCXX_IOSTREAM) && defined(LLS_DEBUG)
    std::cout << "PHI: " << std::endl;
    std::cout << phi.format(fmt) << std::endl;
    // std::cout << "PHI': " << std::endl;
    // std::cout << phi.transpose().format(fmt) << std::endl;
#endif
    computeParams();
}

void PolyLLS::computeParams()
{
    // Update size of theta to match param count
    theta = VectorXd(this->M);

    MatrixXd phiTransposed = this->phi.transpose();
    MatrixXd covar = phiTransposed * this->phi;
    FullPivLU<MatrixXd> lu(covar);

    this->theta = (covar).inverse() * phiTransposed * this->y;
#if defined(_GLIBCXX_IOSTREAM) && defined(LLS_DEBUG)
    std::cout << "Covar Matrix" << std::endl;
    std::cout << (phiTransposed * phi).format(fmt) << std::endl;
    std::cout << "is invertable " << lu.isInvertible() << std::endl;
    std::cout << "Inverted Covar" << std::endl;
    std::cout << (covar).inverse().format(fmt) << std::endl; 
    std::cout << "THETA: " << std::endl;
    std::cout << theta.format(fmt) << std::endl;
#endif
}
