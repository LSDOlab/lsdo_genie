from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import sys

compile_args = []

if sys.platform.startswith('darwin'):
    compile_args=['-std=c++17', '-stdlib=libc++']
else:
    compile_args=['-std=c++17']

list_of_pyx_names = [
    ('core', 'bsplines', 'cython', 'basis0'),
    ('core', 'bsplines', 'cython', 'basis1'),
    ('core', 'bsplines', 'cython', 'basis2'),
    ('core', 'bsplines', 'cython', 'basis_matrix_curve'),
    ('core', 'bsplines', 'cython', 'basis_matrix_curve_py'),
    ('core', 'bsplines', 'cython', 'basis_matrix_surface'),
    ('core', 'bsplines', 'cython', 'basis_matrix_surface_py'),
    ('core', 'bsplines', 'cython', 'basis_matrix_volume'),
    ('core', 'bsplines', 'cython', 'basis_matrix_volume_py'),
]

ext_modules = []
packages=[]
for name_list in list_of_pyx_names:
    ext_name = 'lsdo_genie'
    source_name = 'lsdo_genie'
    packages.append('lsdo_genie.'+'.'.join(name_list[:-1]))
    for name_part in name_list:
        ext_name = '{}.{}'.format(ext_name, name_part)
        source_name = '{}/{}'.format(source_name, name_part)
    source_name = source_name + '.pyx'
    ext_modules = ext_modules + cythonize(
        Extension(
            name=ext_name,
            sources=[source_name],
            language='c++',
            extra_compile_args=compile_args,
            include_dirs=[np.get_include()],
        ),
        annotate=True,
        build_dir='build',
        language_level="2",
    )

# Remove duplicates
packages = list(set(packages))

setup(
    name='lsdo_genie',
    ext_modules=ext_modules,
    version='0.0.0',
    packages=packages,
    install_requires=[
        'numpy',
        'pytest',
        'myst-nb',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi',
        'numpydoc',
        'gitpython',
        'sphinxcontrib-collections @ git+https://github.com/anugrahjo/sphinx-collections.git', # 'sphinx-collections',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine',
        'Cython',
        'numpy-stl',
        'matplotlib',
        'seaborn',
        'scipy',
        'csdl',
    ],
)
