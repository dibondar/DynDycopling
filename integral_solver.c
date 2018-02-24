#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <nlopt.h>

double I1(const double *u, const size_t u_size)
/*
    Evaluate the sum

        I1 = \sum_{k=0}^{u_size-1} sinh(u[k])
*/
{
    double sum = 0.;

    for(size_t k = 0; k < u_size; k++)
        sum += sinh(u[k]);

    return sum;
}

double I2(const double *u, const size_t u_size)
/*
    Evaluate the sum

        I2 = \sum_{k=0}^{u_size-1} \sum_{l=0}^k sinh(u[k] - u[l])
*/
{
    double sum = 0.;

    for(size_t k = 0; k < u_size; k++)
        for(size_t l = 0; l < k + 1; l++)
            sum += sinh(u[k] - u[l]);

    return sum;
}

double J(const double *u, const size_t u_size)
/*
    Evaluate the objective function
*/
{
    return pow(I1(u, u_size), 2) + pow(I2(u, u_size), 2);
}

double nloptJ(unsigned N, const double *u, double *grad_J, void *tmp)
/*
    The function, needed for the NLOPT library, returns the value of the objective function
    and calculates the gradient (grad_J).
*/
{
    // Pre-calculate the sums for I1 and I2
    const double sum_I1 = I1(u, N), sum_I2 = I2(u, N);

    #pragma omp parallel for
    for(size_t n = 0; n < N; n++)
    {
        double sum_pos = 0.;

        for(size_t l = 0; l < n; l++)
            sum_pos += cosh(u[n] - u[l]);

        double sum_neg = 0.;

        for(size_t l = n; l < N; l++)
            sum_neg += cosh(u[n] - u[l]);

        // Save the gradient
        grad_J[n] = 2. * sum_I1 * cosh(u[n]) + 2. * sum_I2 * (sum_pos - sum_neg);
    }

    // return the value of the objective function
    return pow(sum_I1, 2) + pow(sum_I2, 2);
}

double minimizeJ(double *u, const size_t u_size)
/*
    Find u by minimizing J
*/
{
    nlopt_opt opt;

    // Set the algorithm  NLOPT_LD_TNEWTON
    opt = nlopt_create(NLOPT_LD_MMA, u_size);

    // Set the objective function
    nlopt_set_min_objective(opt, nloptJ, NULL);

    // Set the relative numerical tolerance for convergence
    nlopt_set_xtol_rel(opt, 1e-12);

    // Start the minimization
    double minJ = -1.;

    if (nlopt_optimize(opt, u, &minJ) < 0)
        printf("nlopt failed!\n");

    // Clean up
    nlopt_destroy(opt);

    return minJ;
}
