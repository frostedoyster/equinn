#include <torch/extension.h>
#include <cstddef>
#include <iostream> 
#include <cmath>


template<typename scalar_t>
void _sperical_bessel_first_kind_forward(
    int l,
    torch::Tensor x,
    torch::Tensor output
    ) {

    std::size_t total_size = 1;
    for (const auto &size : x.sizes()) {
        total_size *= size;
    }
    
    scalar_t* x_ptr = x.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();

    for (std::size_t index = 0; index < total_size; index++) {
        output_ptr[index] = std::sph_bessel(l, x_ptr[index]);
    }

}


torch::Tensor sperical_bessel_first_kind_forward(
    int l,
    torch::Tensor x
) {

    auto output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "sperical_bessel_first_kind_forward", ([&] {
        _sperical_bessel_first_kind_forward<scalar_t>(
            l,
            x, 
            output
        );
    }));

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("first_kind_forward", &sperical_bessel_first_kind_forward, "spherical Bessel first kind (forward)");
}













/*
int main() {

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