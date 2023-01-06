#include <torch/extension.h>
#include <cstddef>
#include <iostream> 
#include <cmath>


template<typename scalar_t>
void _spherical_bessel_first_kind_forward(
    int l,
    torch::Tensor x,
    torch::Tensor output
    ) {

    int64_t size = x.sizes()[0];

    for (int64_t index = 0; index < size; index++) {
        scalar_t x_item = x.index({index}).item<scalar_t>();
        output.index_put_(
            {index}, 
            std::sph_bessel(l, x_item)
        );
    }

}


torch::Tensor spherical_bessel_first_kind_forward(
    int l,
    torch::Tensor x
) {

    auto output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "spherical_bessel_first_kind_forward", ([&] {
        _spherical_bessel_first_kind_forward<scalar_t>(
            l,
            x, 
            output
        );
    }));

    return output;
}


template<typename scalar_t>
void _spherical_bessel_first_kind_backward(
    int l,
    torch::Tensor x,
    torch::Tensor derivatives
    ) {

    int64_t size = x.sizes()[0];

    for (int64_t index = 0; index < size; index++) {
        scalar_t x_item = x.index({index}).item<scalar_t>();
        derivatives.index_put_(
            {index}, 
            l/x.index({index}).item<scalar_t>() * std::sph_bessel(l, x_item) - std::sph_bessel(l+1, x_item)
        );
    }

}


torch::Tensor spherical_bessel_first_kind_backward(
    int l,
    torch::Tensor x
) {

    auto derivatives = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "spherical_bessel_first_kind_backward", ([&] {
        _spherical_bessel_first_kind_backward<scalar_t>(
            l,
            x, 
            derivatives
        );
    }));

    return derivatives;
}


template<typename scalar_t>
void _spherical_bessel_second_kind_forward(
    int l,
    torch::Tensor x,
    torch::Tensor output
    ) {

    int64_t size = x.sizes()[0];

    for (int64_t index = 0; index < size; index++) {
        scalar_t x_item = x.index({index}).item<scalar_t>();
        output.index_put_(
            {index}, 
            std::sph_neumann(l, x_item)
        );
    }

}


torch::Tensor spherical_bessel_second_kind_forward(
    int l,
    torch::Tensor x
) {

    auto output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "spherical_bessel_second_kind_forward", ([&] {
        _spherical_bessel_second_kind_forward<scalar_t>(
            l,
            x, 
            output
        );
    }));

    return output;
}


template<typename scalar_t>
void _spherical_bessel_second_kind_backward(
    int l,
    torch::Tensor x,
    torch::Tensor derivatives
    ) {

    int64_t size = x.sizes()[0];

    for (int64_t index = 0; index < size; index++) {
        scalar_t x_item = x.index({index}).item<scalar_t>();
        derivatives.index_put_(
            {index}, 
            l/x_item * std::sph_neumann(l, x_item) - std::sph_neumann(l+1, x_item)
        );
    }

}


torch::Tensor spherical_bessel_second_kind_backward(
    int l,
    torch::Tensor x
) {

    auto derivatives = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "spherical_bessel_second_kind_backward", ([&] {
        _spherical_bessel_second_kind_backward<scalar_t>(
            l,
            x, 
            derivatives
        );
    }));

    return derivatives;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("first_kind_forward", &spherical_bessel_first_kind_forward, "spherical Bessel first kind (forward)");
    m.def("first_kind_backward", &spherical_bessel_first_kind_backward, "spherical Bessel first kind (backward)");
    m.def("second_kind_forward", &spherical_bessel_second_kind_forward, "spherical Bessel second kind (forward)");
    m.def("second_kind_backward", &spherical_bessel_second_kind_backward, "spherical Bessel second kind (backward)");
}



/*
int main() {

    // Also works for std::sph_neumann

    int l = 3;
    double x = 3.4;

    // Function
    double f = std::sph_bessel(l, x);
    std::cout << "Value of the function: " << f << std::endl;
    
    // Derivative (exact):
    double df = (l/x) * std::sph_bessel(l, x) - std::sph_bessel(l+1, x);

    // Derivative (finite difference):
    double df_finite = (std::sph_bessel(l, x+0.0001) - std::sph_bessel(l, x-0.0001)) / (2*0.0001);

    std::cout << df << " " << df_finite << std::endl;

    return 0;
}
*/
