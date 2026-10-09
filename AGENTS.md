# AGENTS.md

This project uses a Python environment for development and tests.

## Setting up a development environment

1. Install the project with its test dependencies:
   ```bash
   python -m pip install -e . --group dev
   ```
2. Run the tests:
   ```bash
   pytest
   ```

## Building the documentation

Run the following commands if you want to build the docs locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

## HTTPX2 mocking dependency

The test suite uses `httpx2-pytest>=2` and its `httpx2_mock` fixture and
`IteratorStream` utility. Do **not** install `pytest-httpx2`: it is an
unrelated RESPX-based package which also imports as `pytest_httpx2` and can
silently overwrite the required module, breaking test collection.

If test collection reports that `pytest_httpx2.IteratorStream` cannot be
imported, repair the **same** Python environment used to run pytest:

```powershell
python -m pip uninstall -y pytest-httpx2 httpx2-pytest
python -m pip install "httpx2-pytest>=2"
```
