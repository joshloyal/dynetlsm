#!/sur/bin/env bash
set -x
set -e

MAKE_TARGET=html

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

# install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     -O miniconda.sh
 chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
 export PATH="$MINICONDA_PATH/bin:$PATH"
 conda update --yes --quiet conda

 # Configure the conda environment and put it in the path using the
 # provided versions
 conda create -n $CONDA_ENV_NAME --yes --quiet python=3
 source activate $CONDA_ENV_NAME

 conda install --yes pip numpy scipy scikit-learn pillow matplotlib sphinx \
    numpydoc
pip install -U git+https://github.com/sphinx-gallery/sphinx-gallery.git

# Build and install the docs
cd "$HOME/$CIRCLE_PROJECT_REPONAME"
ls -l
pip install -r doc/requirements.txt
python setup.py clean
python setup.py develop

# the pipefail is requested to propagate the exit code
set -o pipefail && cd doc && make html 2>&1 | tee ~/log.txt

cd -
set +o pipefail
