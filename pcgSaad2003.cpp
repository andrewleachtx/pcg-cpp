#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
using namespace std;
using namespace Eigen;

/*
    Eigen / CPP Implementation of pcgSaad2003
    
    Given Ax = b, iteratively solve for x
    
    optional args:
        - tol: tolerance for convergence
        - maxit: maximum number of iterations
        - M: preconditioner
        - x0: initial guess

    returns:
        - x: solution
        - iter: number of iterations
        - rs: residual vector per iter, default arg is 0x0
*/
// TODO: Rewrite as template to support float/double and streamlined dimensions?
tuple<VectorXf, size_t, vector<float> > pcgSaad2003(const MatrixXf& A, const MatrixXf& b, float tol=1e-6f, int maxit=1000,
                 MatrixXf M=MatrixXf(), VectorXf x0=VectorXf()) {
    
    // We can deduce n from A, assuming it is consistent
    assert(A.rows() == A.cols());
    int n = A.rows();

    assert(n > 0);
    assert(b.rows() == n);
    assert(b.cols() == 1);
    
    // Default M to identity if not passed in
    if (M.rows() == 0) {
        M = MatrixXf::Identity(n, n);
    }

    // If x0 not passed in, default to zero vector
    if (x0.rows() == 0) {
        x0 = VectorXf::Zero(n);
    }

    // Pre-iteration variables
    VectorXf xj = x0;
    VectorXf rj = b - A * xj;

    // Used Cholesky solver to abstract A \ b operator https://eigen.tuxfamily.org/dox/classEigen_1_1LDLT.html#title21
    VectorXf zj = M.ldlt().solve(rj);
    VectorXf pj = zj;
    float tolsq = tol * tol;
    float res1 = rj.dot(rj);
    float res0 = res1;
    vector<float> rs = {res1};

    int i = 0;
    for (i = 0; i < maxit; i++) {
        VectorXf Apj = A * pj;
        float aj = rj.dot(zj) / Apj.dot(pj);
        VectorXf xj1 = xj + aj * pj;
        VectorXf rj1 = rj - aj * Apj;
        VectorXf zj1 = M.ldlt().solve(rj1);
        float bj = rj1.dot(zj1) / rj.dot(zj);
        VectorXf pj1 = zj1 + bj * pj;
        xj = xj1;
        rj = rj1;
        zj = zj1;
        pj = pj1;
        res1 = rj.dot(rj);
        rs.push_back(res1);
        if (res1 < tolsq * res0) {
            break;
        }
    }

    return {xj, i, rs};
}

int main() {
    const int n = 3;
    
    // Test cases comparing to inbuilt solver(s) using random floats. These many not be semi-positive definite, so we enforce
    const MatrixXf A_r = MatrixXf::Random(n, n);
    const VectorXf b_r = VectorXf::Random(n);
    const MatrixXf A = A_r.transpose() * A_r;
    const VectorXf b = A * VectorXf::Random(n);

    cout << "A =" << endl << A << endl
         << "b =" << endl << b << endl;

    // Inbuilt (Eigen::ConjugateGradient)
    ConjugateGradient<MatrixXf> cg;
    cg.compute(A);
    Vector<float, n> x_cgLU = cg.solve(b);
    cout << "Eigen Inbuilt Solver" << endl
         << "x:" << endl << x_cgLU << endl;

    // pcgSaad2003
    auto [x1, iter, rs] = pcgSaad2003(A, b);
    cout << "pcgSaad2003: " << endl << x1 << endl
         << "Iterations: " << iter << endl
         << "Residuals:" << endl;
    // Can't underflow, rs guaranteed size 1
    for (size_t i = 0; i < rs.size() - 1; i++) {
        cout << rs[i] << ", ";
    }
    cout << rs[rs.size() - 1] << endl;

    return 0;
}