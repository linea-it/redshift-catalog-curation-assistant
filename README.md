
# redshift-catalog-curation-assistant

A simple, reproducible and extensible tool to aid in the technical curation of heterogeneous redshift catalogs.

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/redshift-catalog-curation-assistant?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/redshift-catalog-curation-assistant/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/luigilcsilva/redshift-catalog-curation-assistant/smoke-test.yml)](https://github.com/luigilcsilva/redshift-catalog-curation-assistant/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant)
[![Read The Docs](https://img.shields.io/readthedocs/redshift-catalog-curation-assistant)](https://redshift-catalog-curation-assistant.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/luigilcsilva/redshift-catalog-curation-assistant/asv-main.yml?label=benchmarks)](https://luigilcsilva.github.io/redshift-catalog-curation-assistant/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create -n <env_name> python=3.12
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> ./.setup_dev.sh
>> conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
2. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
