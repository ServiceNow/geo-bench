build:
	uv build

release-pypi: build
	uv publish

release-testpypi: build
	uv publish --publish-url https://test.pypi.org/legacy/
