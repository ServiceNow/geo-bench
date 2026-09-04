build:
	python -m build

release-pypi: build
	twine upload dist/*

release-testpypi: build
	twine upload --repository testpypi dist/*
