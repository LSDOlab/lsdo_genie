# lsdo_project_template
A template repository for LSDOlab projects

This repository serves as a template for all LSDOlab projects with regard to documentation, testing and hosting of open-source code.

*README.md file contains high-level information about your package: it's purpose, high-level instructions for installation and usage.*

# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/lsdo_project_template.git
```
If you want users to install a specific branch, run
```sh
pip install git+https://github.com/LSDOlab/lsdo_project_template.git@branch
```

**Enabled by**: `packages=find_packages()` in the `setup.py` file.

## Installation instructions for developers
To install `lsdo_project_template`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/lsdo_project_template.git
pip install -e ./lsdo_project_template
```

# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.
