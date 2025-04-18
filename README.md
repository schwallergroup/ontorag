

<br>
<!--
[![tests](https://github.com/schwallergroup/ontorag/actions/workflows/tests.yml/badge.svg)](https://github.com/schwallergroup/ontorag)
[![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)
[![PyPI](https://img.shields.io/pypi/v/ontorag)](https://img.shields.io/pypi/v/ontorag)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ontorag)](https://img.shields.io/pypi/pyversions/ontorag)
[![Documentation Status](https://readthedocs.org/projects/OntoRAG/badge/?version=latest)](https://OntoRAG.readthedocs.io/en/latest/?badge=latest)
-->

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Cookiecutter template from @SchwallerGroup](https://img.shields.io/badge/Cookiecutter-schwallergroup-blue)](https://github.com/schwallergroup/liac-repo)
[![Learn more @SchwallerGroup](https://img.shields.io/badge/Learn%20%0Amore-schwallergroup-blue)](https://schwallergroup.github.io)


<h1 align="center">
  OntoRAG
</h1>

<p align="center">
  <b>Ontology-based RAG for Scientific Discovery</b>
</p>

<br>


## 🔥 Usage

OntoRAG lets you define a Q&A system that is grounded to predefined ontologies of specific fields.

```python
from OntoRAG.ontorag import OntoRAG

# Initialize OntoRAG with your ontology
orag = OntoRAG(
    ontology_path="path/to/your/ontology",
)

# Ask a question (in the domain of the ontology)
question = "What's the difference between DNA and RNA"
answer = orag.forward(question)
print(answer)
```

You'll need to set up your OpenAI API key in your environment:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or use a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## 👩‍💻 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/OntoRAG/) with:

```shell
$ pip install OntoRAG
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/schwallergroup/ontorag.git
```

## 📂 OntoGen: [Ontogen package](src/OntoGen/)

The OntoGen package is a tool for generating ontologies from a set of text documents. Ontogen is located in ```src/OntoGen``` directory. 

For more details see the [OntoGen README](src/OntoGen/README.md).

## ✅ Citation

Andres M Bran. et al. OntoRAG - Ontology-based RAG for Scientific Discovery
```bibtex
@Misc{ontorag_bran2025,
    author = {Bran, Andres M and Oarga, Alexandru and Hart, Matthew and Lederbauer, Magdalena and Schwaller, Philippe},
    title = {OntoRAG - Ontology-based RAG for Scientific Discovery},
    howpublished = {Github},
    year = {2025},
    url = {https://github.com/schwallergroup/ontorag}
}
```

## 🛠️ For Developers


<details>
  <summary>See developer instructions</summary>



### 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/schwallergroup/ontorag/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.


### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/schwallergroup/ontorag.git
$ cd ontorag
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/schwallergroup/ontorag/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/schwallergroup/ontorag.git
$ cd ontorag
$ tox -e docs
$ open docs/build/html/index.html
```

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/OntoRAG/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.
</details>
