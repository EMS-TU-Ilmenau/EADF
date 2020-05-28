# use 'make target PYTHON=/path/to/python' to specify a
# certain python version to use

# make doc: builds the sphinx docu
# make black: reformats the code
# make all: builds all of the above targets

PYTHON=python3

.PHONY: test
test: | stylecheck
	$(PYTHON) setup.py test

.PHONY: black
black:
	black -l79 demo/*.py eadf/*.py test/*.py *.py doc/source/*.py

.PHONY: doc
doc:
	$(PYTHON) setup.py build_sphinx -E -b html
	$(PYTHON) setup.py build_sphinx -E -b latex
	cd doc &&	make latexpdf LATEXMKOPTS='-shell-escape'
	cp doc/build/latex/eadf.pdf docu.pdf

.PHONY: stylecheck
stylecheck:
	pycodestyle --max-line-length=80 --statistics --ignore=E203,W503 eadf/*.py test/*.py demo/*.py *.py

all: black doc
