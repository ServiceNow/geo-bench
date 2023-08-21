release-pypi:
	poetry build
	poetry publish --repository pypi

release-testpypi:
	poetry build
	poetry publish --repository testpypi