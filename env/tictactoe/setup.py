from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = [
    Extension("tictactoeboard",
        sources=["tictactoeboard.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17","-fopenmp"],
        extra_link_args=["-fopenmp"])]

setup(
    name = "tictactoeboard",
    ext_modules = cythonize(extensions)
)