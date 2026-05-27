# Contributing to SynaptoRoute

First off, thank you for considering contributing to SynaptoRoute! It's people like you that make open source such a great community.

## Development Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/SynaptoRoute.git
   cd SynaptoRoute
   ```
3. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -e .[api]
   pip install pytest
   ```

## Running Tests
We enforce strict testing for all architectural components (encoding, latency, routing accuracy). Before submitting a PR, ensure all tests pass:
```bash
pytest tests/
```

## Pull Request Process
1. Ensure your code strictly adheres to the existing architectural philosophy (e.g., preserving $O(1)$ Lazy Compilation).
2. Update the `README.md` or Jupyter Notebooks in `notebooks/` if you add new features.
3. Open a Pull Request using the provided GitHub PR template.
4. Wait for CI/CD checks to pass.
