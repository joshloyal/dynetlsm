# makefile to simplify repetitive build env managment tasks under posix

PYTHON ?= python
PYTEST ?= pytest

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

install-dev:
	$(PYTHON) setup.py develop

test-code: install-dev
	$(PYTEST) --showlocals -v sliced
