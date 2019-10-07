/*
 * NNLS_all_inds_dome_subproblem_mex:
 *
 * 	Solves the feature elimination "dome" subproblem
 *
 * 		min 	<a, nu>
 * 		s.t.	||nu - nu0||^2 <= 2*gap_bound
 * 				<a_i, nu> >= 0
 *
 * 	repeatedly, for each column a of A, and for each column a_i of A.
 *
 *  This is used for NNLS feature elimination:
 *
 *       Strong duality ball + single dual feasibility constraint
 *       but we try all singleton dual feasibility constraints and take the
 *       max bound that we find.
 *         max_i   min_nu  <a, nu>
 *                 s.t.    ||nu - nu0||^2 <= 2*gap_bound
 *                         <a_i, nu> >= 0
 *
 * 	For each a in A, this returns the maximum optimal value over all a_i,
 * 	ignoring any infeasible problems (by setting the optimal value to -inf).
 * 	The output is a matrix of maximum optimal values of the same shape as nu0.
 *
 * 	Since this will be used for feature elimination problems, this returns
 * 	val = -inf if all problems are infeasible (the standard thing to do is val = +inf).
 *
 * Inputs:
 *	A
 *	nu0
 *	gap_bound
 *
 * Outputs:
 * 	bounds
 *
 */

#include <stdexcept>
#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

#include <omp.h>

#include <mex.h>

#include <blas.h>


class AllIndsDomeSubproblems {
public:
	AllIndsDomeSubproblems(ptrdiff_t m,
						   ptrdiff_t n,
						   ptrdiff_t n_rhs,
						   double *A,
						   double *nu0,
						   double *gap_bound) :
		m_(m), n_(n), n_rhs_(n_rhs), A_(A), nu0_(nu0) {

		for (ptrdiff_t i=0; i<n_rhs_; ++i) {
			if (gap_bound[i] < 0.) {
				throw std::invalid_argument("AllIndsDomeSubproblems: gap_bound should be nonnegative.");
			}
		}

		gap_bound_ = gap_bound;

		init_();
	}

	double bound(ptrdiff_t rhs, ptrdiff_t feat, ptrdiff_t i) {
		double dot_anu0 = dot_anu0_[rhs*n_ + feat];
		double dot_ainu0 = dot_anu0_[rhs*n_ + i];

		if (gap_bound_[rhs] == 0.) {
			if (dot_ainu0 >= 0.) {
				return dot_anu0;
			} else { // not necessarily infeasible, but for convenience we return -inf
				return -std::numeric_limits<double>::infinity();
			}
		}

		double norm_a2 = norm_a_[feat]*norm_a_[feat];
		double norm_ai2 = norm_a_[i]*norm_a_[i];
		double dot_aai = dot_aai_[feat*n_ + i];
		double delta = 2*gap_bound_[rhs];

		double condition = (dot_aai - 2.*dot_ainu0*norm_a_[feat]/(2*std::sqrt(delta)))/norm_ai2;

		if (condition > 0) { // lambda2 > 0
			double tmp = (dot_aai*dot_aai/norm_ai2 - norm_a2);
			tmp /= -4*dot_ainu0*dot_ainu0/norm_ai2 - 4*delta;

			if (tmp < 0) { // we relax some of the checks and just bail; this is weaker, but valid.
				return -std::numeric_limits<double>::infinity();
			}

			double lambda1 = std::sqrt(tmp);
			double lambda2 = (dot_aai - 2*lambda1*dot_ainu0)/norm_ai2;

			double val = dot_anu0 - lambda1*delta - lambda2*dot_ainu0;
			val += -1/(4*lambda1)*(lambda2*lambda2*norm_ai2 - 2*lambda2*dot_aai + norm_a2);
			return val;

		} else {
			return dot_anu0 - norm_a_[feat]*std::sqrt(delta);

		}
	}

	void computeBounds(double *bounds) {
		for (ptrdiff_t i=0; i<n_*n_rhs_; ++i) {
			bounds[i] = -std::numeric_limits<double>::infinity();
		}

		#pragma omp parallel for num_threads(omp_get_max_threads())
		for (ptrdiff_t rhs=0; rhs<n_rhs_; ++rhs) {
			//std::cout << "\rAllIndsDomeSubproblems::computeBounds: rhs = " << rhs+1 << std::flush;
			for (ptrdiff_t feat=0; feat<n_; ++feat) {
				ptrdiff_t rhsn = rhs*n_;
				for (ptrdiff_t i=0; i<n_; ++i) {
					bounds[rhsn + feat] = std::max(bounds[rhsn + feat], bound(rhs, feat, i));
				}
			}
		}
		//std::cout << std::endl;
	}

private:

	void init_(void) {
		constexpr double done = 1.;
		constexpr double dzero = 0.;

		dot_anu0_.resize(n_*n_rhs_);
		dot_aai_.resize(n_*n_);
		norm_a_.resize(n_);

		// dot_anu0_
		dgemm("T", "N", &n_, &n_rhs_, &m_, &done, A_, &m_, nu0_, &m_, &dzero, &dot_anu0_[0], &n_);

		// dot_aai_
		dgemm("T", "N", &n_, &n_, &m_, &done, A_, &m_, A_, &m_, &dzero, &dot_aai_[0], &n_);

		// norm_a_
		for (ptrdiff_t i=0; i<n_; ++i) {
			norm_a_[i] = std::sqrt(dot_aai_[i*n_ + i]);
		}
	}

	ptrdiff_t m_, n_, n_rhs_;
	double *A_, *nu0_, *gap_bound_;
	std::vector<double> dot_anu0_, dot_aai_, norm_a_;
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// Check number of input/output arguments
	if (nrhs != 3) {
		throw std::runtime_error("NNLS_all_inds_dome_subproblem_mex: should have exactly 3 inputs.");
	}

	if (nlhs > 1) {
		throw std::runtime_error("NNLS_all_inds_dome_subproblem_mex: should have no more than 1 output.");
	}

	// Check input sizes
	size_t m = mxGetM(prhs[0]);
	size_t n = mxGetN(prhs[0]);
	size_t n_rhs = mxGetN(prhs[1]);

	if (mxGetM(prhs[1]) != m) {
		throw std::runtime_error("NNLS_all_inds_dome_subproblem_mex: nu0 should be an [m n_rhs] matrix.");
	}

	if (mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != n_rhs) {
		throw std::runtime_error("NNLS_all_inds_dome_subproblem_mex: gap_bound should be an [1 n_rhs] vector.");
	}

	// Get input data
	double *A = mxGetPr(prhs[0]);
	double *nu0 = mxGetPr(prhs[1]);
	double *gap_bound = mxGetPr(prhs[2]);

	if (nlhs >= 1) {
		// Initialize output data
		plhs[0] = mxCreateDoubleMatrix(n, n_rhs, mxREAL);
		double *bounds = mxGetPr(plhs[0]);

		// Do the stuff
		AllIndsDomeSubproblems all_inds_dome_subproblems = AllIndsDomeSubproblems(m, n, n_rhs, A, nu0, gap_bound);
		all_inds_dome_subproblems.computeBounds(bounds);

	}
}

