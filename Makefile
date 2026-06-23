# SuperLocalMemory Makefile
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar

.PHONY: test test-slow lint build publish publish-npm clean

# GC-safe test run (macOS ARM: prevents libomp/tokio SIGSEGV)
# -p no:cacheprovider  — no .pytest_cache writes
# -p no:subtests       — disable pytest-subtests reporter that caused SIGSEGV
# PYTHONMALLOC=malloc  — system allocator (no jemalloc/tcmalloc fork hazard)
test:
	env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
	    VECLIB_MAXIMUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONMALLOC=malloc \
	    PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider -p no:subtests \
	    -q -o faulthandler_timeout=0 \
	    -m "not slow and not ollama and not benchmark" \
	    $(ARGS)

test-slow:
	env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
	    VECLIB_MAXIMUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONMALLOC=malloc \
	    PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider -p no:subtests \
	    -q -o faulthandler_timeout=0 \
	    $(ARGS)

lint:
	.venv/bin/python -m ruff check src/ tests/
	.venv/bin/python -m ruff format --check src/ tests/

build:
	.venv/bin/python -m build

# PyPI publish — requires PYPI_TOKEN env var
# NEVER run as npm postinstall. Manual step only.
publish:
	@if [ -z "$$PYPI_TOKEN" ]; then \
	    echo "ERROR: PYPI_TOKEN not set. Load from ~/.claude-secrets.env"; \
	    exit 1; \
	fi
	.venv/bin/python -m build
	TWINE_USERNAME=__token__ TWINE_PASSWORD="$$PYPI_TOKEN" \
	    .venv/bin/twine upload dist/superlocalmemory-$(shell grep '^version' pyproject.toml | cut -d'"' -f2)*

# npm publish — requires NPM_TOKEN env var
# Uses temp npmrc to avoid writing token to ~/.npmrc permanently.
publish-npm:
	@if [ -z "$$NPM_TOKEN" ]; then \
	    echo "ERROR: NPM_TOKEN not set. Load from ~/.claude-secrets.env"; \
	    exit 1; \
	fi
	@NPMRC_TMP=$$(mktemp); \
	    echo "//registry.npmjs.org/:_authToken=$$NPM_TOKEN" > "$$NPMRC_TMP"; \
	    npm publish --userconfig "$$NPMRC_TMP" && rm -f "$$NPMRC_TMP" || (rm -f "$$NPMRC_TMP"; exit 1)

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ src/superlocalmemory.egg-info/ .pytest_cache/ .pytest_tmp_data/
