# Contributing to LoPace

Thank you for your interest in contributing to LoPace! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LoPace.git
   cd LoPace
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/amanulla/lopace.git
   ```

4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   pytest tests/ -v
   ```

## Making Changes

### Types of Contributions

We welcome various types of contributions:

- **Bug Fixes**: Fix issues in existing code
- **New Features**: Add new compression methods or functionality
- **Documentation**: Improve README, docstrings, or guides
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add usage examples or demos

### Before Making Changes

1. **Check existing issues** to see if someone is already working on it
2. **Open an issue** if you're planning a significant change to discuss the approach
3. **Keep changes focused** - one feature or fix per pull request

## Code Style

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Keep lines under 127 characters
- Use meaningful variable and function names

### Formatting

We use `black` for code formatting:

```bash
black lopace tests
```

### Linting

We use `flake8` for linting:

```bash
flake8 lopace tests
```

### Docstrings

Follow Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Example:
        >>> example_function("hello", 42)
        True
    """
    pass
```

## Testing

### Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=lopace --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_compressor.py -v
```

### Writing Tests

- Write tests for all new features
- Aim for high test coverage (>80%)
- Follow the existing test structure
- Use descriptive test names
- Test edge cases and error conditions

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    compressor = PromptCompressor()
    
    # Act
    result = compressor.some_method("input")
    
    # Assert
    assert result == expected_output
```

## Documentation

### Docstrings

- Add docstrings to all public functions and classes
- Include parameter descriptions
- Include return value descriptions
- Add usage examples where helpful

### README Updates

- Update README.md if adding new features
- Add usage examples for new functionality
- Update installation instructions if dependencies change

### Changelog

- Add entries to [CHANGELOG.md](CHANGELOG.md) for user-facing changes
- Follow the existing format
- Include your PR number

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
feat: Add new compression method for X

Description of what and why (not how)

Fixes #123
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

3. **Run linting**:
   ```bash
   flake8 lopace tests
   black lopace tests
   ```

4. **Update documentation** if needed

5. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request** on GitHub:
   - Fill out the PR template
   - Reference related issues
   - Describe your changes
   - Include screenshots for UI changes

7. **Wait for review**:
   - Address reviewer comments
   - Keep discussions constructive
   - Update PR based on feedback

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No merge conflicts
- [ ] All CI checks passing

## Review Process

- PRs require at least one approval
- Maintainers will review within 48 hours
- Address feedback promptly
- Be open to suggestions and improvements

## Getting Help

- **Questions?** Open a [Discussion](https://github.com/amanulla/lopace/discussions)
- **Found a bug?** Open an [Issue](https://github.com/amanulla/lopace/issues)
- **Security issue?** Email the maintainers (see SECURITY.md)

## Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Credited in the project

Thank you for contributing to LoPace! 🎉