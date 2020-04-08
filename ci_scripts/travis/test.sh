set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR
cp setup.cfg $TEST_DIR
cd $TEST_DIR

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [[ "$COVERAGE" == "true" ]]; then
    pytest --cov=$MODULE --pep8 --pyargs
else
    pytest --pep8 --pyargs
fi
