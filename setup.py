from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

spherical_bessel = cpp_extension.CppExtension('spherical_bessel', 
      ["spherical_bessel.cc"], 
      extra_compile_args=["-std=c++17", "-fopenmp", "-w"])

ext_modules = [spherical_bessel]

setup(name='equinn',
      packages = find_packages(),
      ext_modules = ext_modules,
      cmdclass={'build_ext': cpp_extension.BuildExtension})