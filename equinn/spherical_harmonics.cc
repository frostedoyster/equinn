#include <torch/extension.h>
#include <cstddef>
#include <iostream>
#include <cmath>
#include <vector>


double factorial(unsigned int n) {
    return std::tgamma(n+1);
}


template<typename scalar_t>
std::vector<std::vector<scalar_t>> associated_legendre_polynomials(int l_max, scalar_t x) {

    // Initialize vectors to store the polynomials
    std::vector<std::vector<scalar_t>> p;
    for (int l = 0; l <= l_max; l++) {
        p.push_back(std::vector<scalar_t>(l+1, 0.0));
    }

    p[0][0] = 1.0;
    for (int m = 1; m <= l_max; m++) {
        p[m][m] = -(2*m-1)*std::sqrt((1.0-x)*(1.0+x))*p[m-1][m-1]; 
    }
    for (int m = 0; m < l_max; m++) {
        p[m+1][m] = (2*m+1)*x*p[m][m]; 
    }
    for (int m = 0; m < l_max-1; m++) {
        for (int l = m+2; l <= l_max; l++) {
            p[l][m] = ((2*l-1)*x*p[l-1][m]-(l+m-1)*p[l-2][m])/(l-m);
        } 
    }

    return p;
}


template<typename scalar_t>
std::vector<std::vector<scalar_t>> associated_legendre_polynomial_derivatives(int l_max, scalar_t x, std::vector<std::vector<scalar_t>> p) {

    // Initialize a vector to store the derivatives
    std::vector<std::vector<scalar_t>> dPlm_dx;
    for (int l = 0; l <= l_max; l++) {
        dPlm_dx.push_back(std::vector<scalar_t>(l+1, 0.0));
    }

    // p has to be provided from outside (it is needed anyway for the derivatives of the spherical harmonics w.r.t. phi)
    
    dPlm_dx[0][0] = 0.0;
    // Iterate over the degree of the polynomial
    for (int l = 1; l <= l_max; l++) {
        // Iterate over the orders
        for (int m = 0; m <= l-1; m++) {
            dPlm_dx[l][m] = (x*l*p[l][m] - (m + l) * p[l-1][m]) / (x*x-1);
        }
        dPlm_dx[l][l] = x*l*p[l][l] / (x*x-1);
    }
    
    // Return the derivatives
    return dPlm_dx;
}


template<typename scalar_t>
void _spherical_harmonics_forward(
    int l_max,
    torch::Tensor cos_theta,
    torch::Tensor phi,
    std::vector<torch::Tensor> output
    ) {

    std::size_t input_size = cos_theta.sizes()[0];
    
    scalar_t* cos_theta_ptr = cos_theta.data_ptr<scalar_t>();
    scalar_t* phi_ptr = phi.data_ptr<scalar_t>();
    std::vector<scalar_t*> output_ptrs; 
    
    for (const auto &tensor : output) {
        output_ptrs.push_back(tensor.data_ptr<scalar_t>());
    }

    for (std::size_t index = 0; index < input_size; index++) {
        scalar_t cost = cos_theta_ptr[index];
        scalar_t ph = phi_ptr[index];

        std::vector<std::vector<scalar_t>> plm = associated_legendre_polynomials(l_max, cost);

        for (int l=0; l<=l_max; l++) {
            for (int m=-l; m<=l; m++) {
                if (m > 0) {
                    output_ptrs[l][index*(2*l+1)+m+l] =
                        std::pow(-1, m) * std::sqrt(2.0) 
                        * std::sqrt((2*l+1)/(4*M_PI)*factorial(l-m)/factorial(l+m))
                        * plm[l][m]
                        * std::cos(m*ph);
                } else if (m < 0) {
                    output_ptrs[l][index*(2*l+1)+m+l] =
                        std::pow(-1, m) * std::sqrt(2.0) 
                        * std::sqrt((2*l+1)/(4*M_PI)*factorial(l+m)/factorial(l-m))
                        * plm[l][-m]
                        * std::sin(-m*ph);
                } else {  // m == 0
                    output_ptrs[l][index*(2*l+1)+m+l] = 
                        std::sqrt((2*l+1)/(4*M_PI))
                        * plm[l][m];
                }
            }
        }
    }

}


std::vector<torch::Tensor> spherical_harmonics_forward(
    int l_max,
    torch::Tensor cos_theta,
    torch::Tensor phi
) {

    // Initialize output tensors:
    std::vector<torch::Tensor> output;
    for (int l=0; l<=l_max; l++) {
        output.push_back(
            torch::empty({cos_theta.sizes()[0], 2*l+1}, cos_theta.options())
        );
    }

    AT_DISPATCH_FLOATING_TYPES(cos_theta.type(), "spherical_harmonics_forward", ([&] {
        _spherical_harmonics_forward<scalar_t>(
            l_max,
            cos_theta,
            phi, 
            output
        );
    }));

    return output;
}


template<typename scalar_t>
void _spherical_harmonics_backward(
    int l_max,
    torch::Tensor cos_theta,
    torch::Tensor phi,
    std::vector<torch::Tensor> d_output_d_cos_theta,
    std::vector<torch::Tensor> d_output_d_phi
    ) {

    std::size_t input_size = cos_theta.sizes()[0];
    
    scalar_t* cos_theta_ptr = cos_theta.data_ptr<scalar_t>();
    scalar_t* phi_ptr = phi.data_ptr<scalar_t>();
    std::vector<scalar_t*> d_output_d_cos_theta_ptrs;
    std::vector<scalar_t*> d_output_d_phi_ptrs;
    
    for (const auto &tensor : d_output_d_cos_theta) {
        d_output_d_cos_theta_ptrs.push_back(tensor.data_ptr<scalar_t>());
    }
    for (const auto &tensor : d_output_d_phi) {
        d_output_d_phi_ptrs.push_back(tensor.data_ptr<scalar_t>());
    }

    for (std::size_t index = 0; index < input_size; index++) {
        scalar_t cost = cos_theta_ptr[index];
        scalar_t ph = phi_ptr[index];

        std::vector<std::vector<scalar_t>> plm = associated_legendre_polynomials(l_max, cost);  // Could these be carried over from the forward pass?
        std::vector<std::vector<scalar_t>> d_plm_d_cost = associated_legendre_polynomial_derivatives(l_max, cost, plm);

        for (int l=0; l<=l_max; l++) {
            for (int m=-l; m<=l; m++) {
                if (m > 0) {
                    d_output_d_cos_theta_ptrs[l][index*(2*l+1)+m+l] =
                        std::pow(-1, m) * std::sqrt(2.0) 
                        * std::sqrt((2*l+1)/(4*M_PI)*factorial(l-m)/factorial(l+m))
                        * d_plm_d_cost[l][m]
                        * std::cos(m*ph);
                    d_output_d_phi_ptrs[l][index*(2*l+1)+m+l] =
                        std::pow(-1, m) * std::sqrt(2.0) 
                        * std::sqrt((2*l+1)/(4*M_PI)*factorial(l-m)/factorial(l+m))
                        * plm[l][m]
                        * (-std::sin(m*ph));
                } else if (m < 0) {
                    d_output_d_cos_theta_ptrs[l][index*(2*l+1)+m+l] =
                        std::pow(-1, m) * std::sqrt(2.0) 
                        * std::sqrt((2*l+1)/(4*M_PI)*factorial(l+m)/factorial(l-m))
                        * d_plm_d_cost[l][-m]
                        * std::sin(-m*ph);
                    d_output_d_phi_ptrs[l][index*(2*l+1)+m+l] =
                        std::pow(-1, m) * std::sqrt(2.0) 
                        * std::sqrt((2*l+1)/(4*M_PI)*factorial(l+m)/factorial(l-m))
                        * plm[l][-m]
                        * std::cos(-m*ph);
                } else {  // m == 0
                    d_output_d_cos_theta_ptrs[l][index*(2*l+1)+m+l] = 
                        std::sqrt((2*l+1)/(4*M_PI))
                        * d_plm_d_cost[l][m];
                    d_output_d_phi_ptrs[l][index*(2*l+1)+m+l] = 0.0;
                }
            }
        }
    }

}


std::vector<std::vector<torch::Tensor>> spherical_harmonics_backward(
    int l_max,
    torch::Tensor cos_theta,
    torch::Tensor phi
) {

    // Initialize output tensors:
    std::vector<torch::Tensor> d_output_d_cos_theta;
    std::vector<torch::Tensor> d_output_d_phi;
    for (int l=0; l<=l_max; l++) {
        d_output_d_cos_theta.push_back(
            torch::empty({cos_theta.sizes()[0], 2*l+1}, cos_theta.options())
        );
        d_output_d_phi.push_back(
            torch::empty({cos_theta.sizes()[0], 2*l+1}, phi.options())
        );
    }

    AT_DISPATCH_FLOATING_TYPES(cos_theta.type(), "spherical_harmonics_backward", ([&] {
        _spherical_harmonics_backward<scalar_t>(
            l_max,
            cos_theta,
            phi, 
            d_output_d_cos_theta,
            d_output_d_phi
        );
    }));

    return {d_output_d_cos_theta, d_output_d_phi};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spherical_harmonics_forward, "spherical harmonics (forward)");
    m.def("backward", &spherical_harmonics_backward, "spherical harmonics (backward)");
}


/*
int main() {

    int l_max = 4;
    double x = 0.74;

    // Function
    std::vector<std::vector<double>> f = associated_legendre_polynomials(l_max, x);
    for (std::vector<double> f_l : f) {
        for (double f_lm : f_l) {
            std::cout << f_lm << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;
    
    // Derivative (exact):
    std::vector<std::vector<double>> df = associated_legendre_polynomial_derivatives(l_max, x);
    for (std::vector<double> f_l : df) {
        for (double f_lm : f_l) {
            std::cout << f_lm << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;
    
    // Derivative (finite differences):
    std::vector<std::vector<double>> fp = associated_legendre_polynomials(l_max, x+0.0001);
    std::vector<std::vector<double>> fm = associated_legendre_polynomials(l_max, x-0.0001);
    for (int l=0; l<=l_max; l++) {
        std::vector<double> fp_l = fp[l];
        std::vector<double> fm_l = fm[l];
        for (int m=0; m<=l; m++) {
            std::cout << (fp_l[m] - fm_l[m])/(0.0001*2) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/
