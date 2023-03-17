from setuptools import setup, find_packages

setup(
    name='lsdo_genie',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'myst-nb',
        'sphinx_rtd_theme',
        'sphinx-autoapi',
        'numpydoc',
        'gitpython',
        'sphinx-collections',
        'scipy',
        'pint',
        'sphinx-code-include',
        'jupyter-sphinx',
    ],
)
