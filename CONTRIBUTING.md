# Contributing to SuperLocalMemory V3

Thank you for considering contributing to SuperLocalMemory V3! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Areas for Contribution](#areas-for-contribution)
- [Community](#community)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone, regardless of background or identity.

### Standards

**Expected behavior:**
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

**Unacceptable behavior:**
- Harassment, trolling, or insulting comments
- Personal or political attacks
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying standards and will take appropriate corrective action in response to unacceptable behavior.

Report violations to: [admin@superlocalmemory.com]

---

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

1. **Bug Reports** - Found a bug? Let us know!
2. **Feature Requests** - Have an idea? We'd love to hear it!
3. **Code Contributions** - Submit patches and new features
4. **Documentation** - Improve guides, add examples, fix typos
5. **Testing** - Write tests, perform QA, report edge cases
6. **Design** - UI/UX improvements, diagrams, visualizations

### First Time Contributors

Look for issues labeled:
- `good first issue` - Simple, well-defined tasks
- `documentation` - Documentation improvements
- `help wanted` - We need assistance on these

---

## Development Setup

### Prerequisites

- Python 3.8 or higher
- SQLite3 (usually pre-installed)
- Git
- Text editor or IDE (VS Code, PyCharm, etc.)

### Step 1: Fork and Clone

```bash
# Fork the repository on GitHub (click "Fork" button)

# Clone your fork
git clone https://github.com/YOUR_USERNAME/superlocalmemory.git
cd superlocalmemory-repo

# Add upstream remote
git remote add upstream https://github.com/qualixar/superlocalmemory.git
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Development Dependencies

```bash
# Install project (no external dependencies for core)
# For development/testing:
pip install pytest pytest-cov black flake8

# Optional: Install optional dependencies
pip install scikit-learn leidenalg
```

### Step 4: Verify Setup

```bash
# Run tests
pytest tests/

# Check code style
flake8 src/
black --check src/

# Run installation
# On macOS/Linux:
./install.sh
# Activate virtual environment

# On Windows:
./install.ps1

# Test basic functionality
memory-status
```

### Step 5: Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

---

## Project Structure

```
superlocalmemory/
│
├── src/superlocalmemory/           # Python package source
│   ├── cli/                        # CLI commands
│   │   ├── commands.py                # All CLI commands
│   │   ├── main.py                   # CLI entry point
│   │   ├── daemon.py                 # Background worker
│   │   ├── setup_wizard.py            # Interactive setup
│   │   └── migrate_cmd.py            # V2→V3 migration
│   ├── core/                       # Core engine
│   ├── storage/                    # Database layer
│   ├── retrieval/                  # 6-channel retrieval
│   ├── dynamics/                   # Langevin dynamics
│   ├── math/                       # Fisher-Rao, sheaf cohomology
│   ├── encoding/                   # Embeddings & quantization
│   ├── trust/                      # Trust scoring
│   ├── compliance/                 # EU AI Act, GDPR tools
│   ├── learning/                   # LightGBM pattern learning
│   ├── hooks/                      # Claude Code hooks
│   ├── mcp/                        # MCP server & tools
│   ├── server/                     # Dashboard API server
│   └── ui/                        # Dashboard frontend
│
├── bin/                          # CLI wrappers
│   ├── slm                         # Unix bash wrapper
│   ├── slm-npm                     # Node.js wrapper
│   ├── slm.bat                     # Windows batch
│   └── slm.cmd                    # Windows cmd
│
├── tests/                        # Test suite (1400+ tests)
│
├── docs/                         # Documentation
│
├── ui/                           # Dashboard UI
├── ide/                          # IDE integrations
├── integrations/                  # Third-party integrations
├── scripts/                      # Build & install scripts
├── package.json                  # npm package config
├── pyproject.toml                # Python package config
├── README.md                     # Project overview
├── CONTRIBUTING.md               # This file
├── CHANGELOG.md                  # Version history
├── LICENSE                       # Elastic License 2.0
└── SECURITY.md                   # Security policy
```

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

**Line Length:**
- Maximum 100 characters (vs. PEP 8's 79)
- Break long lines at logical points

**Naming Conventions:**
```python
# Functions and variables: snake_case
def calculate_confidence_score():
    user_preference = "React"

# Classes: PascalCase
class PatternLearner:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CLUSTER_SIZE = 100

# Private members: leading underscore
def _internal_method():
    pass
```

**Docstrings:**

Use Google-style docstrings:

```python
def build_graph(memories, resolution=1.0):
    """Build knowledge graph from memories using Leiden clustering.

    Args:
        memories (list): List of memory dictionaries
        resolution (float): Leiden algorithm resolution parameter

    Returns:
        dict: Graph statistics including cluster count

    Raises:
        ValueError: If memories list is empty

    Example:
        >>> stats = build_graph(memories, resolution=1.2)
        >>> print(stats['cluster_count'])
        5
    """
    pass
```

**Imports:**

Group and order imports:

```python
# 1. Standard library
import json
import sqlite3
from datetime import datetime

# 2. Third-party (if any)
import leidenalg

# 3. Local modules
from memory_store_v2 import MemoryStore
```

### Code Quality Tools

**Black** - Auto-formatter:
```bash
# Format code
black src/

# Check formatting
black --check src/
```

**Flake8** - Linter:
```bash
# Check code style
flake8 src/ --max-line-length=100
```

**Type Hints** (encouraged but not required):
```python
def add_memory(content: str, tags: list[str] = None) -> int:
    """Add memory and return ID."""
    pass
```

---

## Testing Requirements

### Test Framework

We use **pytest** for testing.

### Writing Tests

**Location:** `tests/test_<module_name>.py`

**Example test file:**

```python
# tests/test_memory_store.py

import pytest
import tempfile
import os
from src.memory_store_v2 import MemoryStore

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    yield path
    os.close(fd)
    os.unlink(path)

def test_add_memory(temp_db):
    """Test adding a memory."""
    store = MemoryStore(temp_db)
    memory_id = store.add("Test memory", tags=["test"])

    assert memory_id > 0

    # Verify memory was added
    results = store.search("Test memory")
    assert len(results) == 1
    assert results[0]['content'] == "Test memory"

def test_search_with_tags(temp_db):
    """Test searching by tags."""
    store = MemoryStore(temp_db)
    store.add("React memory", tags=["react", "frontend"])
    store.add("Python memory", tags=["python", "backend"])

    results = store.search_by_tag("react")
    assert len(results) == 1
    assert "React" in results[0]['content']
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_memory_store.py

# Run specific test
pytest tests/test_memory_store.py::test_add_memory

# Verbose output
pytest -v tests/
```

### Test Requirements

**All code contributions must:**
1. Include tests for new functionality
2. Maintain or improve code coverage (aim for 80%+)
3. Pass all existing tests
4. Not break backward compatibility (unless documented)

**For bug fixes:**
- Add a test that reproduces the bug
- Verify test fails before fix
- Verify test passes after fix

---

## Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines (PEP 8, Black formatted)
- [ ] All tests pass (`pytest tests/`)
- [ ] New functionality includes tests
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] Changelog updated (for significant changes)

### Step 1: Update Your Branch

```bash
# Fetch latest changes
git fetch upstream

# Rebase on upstream main
git rebase upstream/main

# Resolve conflicts if any
# Then: git rebase --continue
```

### Step 2: Run Pre-Submit Checks

```bash
# Format code
black src/

# Lint
flake8 src/ --max-line-length=100

# Test
pytest tests/

# Verify installation
./install.sh
memory-status
```

### Step 3: Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: intelligent cluster auto-naming

- Implement TF-IDF term extraction
- Add cluster naming heuristics
- Include tests for edge cases
- Update documentation

Closes #123"
```

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting)
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance tasks

### Step 4: Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
# Fill out PR template
```

### Step 5: PR Review Process

**What happens next:**
1. Automated checks run (tests, linting)
2. Maintainers review code
3. Feedback provided (if needed)
4. Iterate on feedback
5. Approval and merge

**Review criteria:**
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Backward compatibility

---

## Issue Guidelines

### Reporting Bugs

**Use the bug report template:**

```markdown
**Describe the bug**
Clear and concise description of the bug.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected behavior**
What should happen.

**Actual behavior**
What actually happens.

**Environment:**
- OS: [e.g., macOS 12.5]
- Python version: [e.g., 3.9.7]
- SuperLocalMemory version: [e.g., 2.0.0]

**Additional context**
Error messages, logs, screenshots.
```

### Requesting Features

**Use the feature request template:**

```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other approaches you've thought about.

**Additional context**
Use cases, examples, mockups.
```

### Asking Questions

For questions, use:
- GitHub Discussions (preferred)
- Stack Overflow with tag `superlocalmemorv2`
- Issues with label `question`

---

## Areas for Contribution

### High Priority

1. **Performance Optimization**
   - Faster graph builds for 1000+ memories
   - Incremental graph updates (vs. full rebuild)
   - Memory-efficient pattern learning

2. **Graph Visualization**
   - Web-based graph viewer
   - Cluster relationship diagrams
   - Interactive exploration tools

3. **Additional Pattern Categories**
   - Testing preferences (pytest, jest, TDD)
   - DevOps tools (Docker, K8s, CI/CD)
   - Documentation style

4. **Enhanced Search**
   - Semantic search using embeddings
   - Multi-query expansion
   - Search result ranking

### Medium Priority

5. **Integration Plugins**
   - VS Code extension
   - Obsidian plugin
   - Roam Research integration

6. **Export/Import**
   - Markdown export
   - JSON/CSV export
   - Cross-profile memory transfer

7. **Analytics Dashboard**
   - Memory growth over time
   - Topic distribution
   - Learning trajectory visualization

### Nice to Have

8. **Mobile Companion**
   - iOS/Android app for memory capture
   - Voice-to-text memory input

9. **Collaborative Features**
   - Shared knowledge bases (opt-in)
   - Team memory spaces

10. **Advanced ML**
    - Local LLM integration for summarization
    - Automatic tag suggestion

---

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas, show-and-tell
- **Discord:** Real-time chat (coming soon)

### Getting Help

**For contributors:**
- Tag maintainers in PR comments
- Ask in GitHub Discussions
- Check existing issues and PRs

**For users:**
- Check documentation first
- Search existing issues
- Create new issue with detailed information

---

## Recognition

### Contributors

All contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in documentation

### Maintainers

Current maintainers:
- [List of core maintainers]

Maintainers are responsible for:
- Code review
- Issue triage
- Release management
- Community moderation

---

## License

By contributing to SuperLocalMemory V3, you agree that your contributions will be licensed under the Elastic License 2.0.

See [LICENSE](LICENSE) for details.

---

## Additional Resources

### Learning Resources

**Python Development:**
- [PEP 8 Style Guide](https://pep8.org/)
- [Real Python](https://realpython.com/)
- [pytest Documentation](https://docs.pytest.org/)

**Knowledge Graphs:**
- [GraphRAG Paper](https://example.com)
- [Leiden Algorithm](https://example.com)
- [TF-IDF Explained](https://example.com)

**SQLite:**
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [FTS5 Full-Text Search](https://www.sqlite.org/fts5.html)

### Project Documentation

- [Architecture](ARCHITECTURE.md) - Technical design
- [Installation](INSTALL.md) - Setup guide
- [Quick Start](QUICKSTART.md) - First steps
- [Security](SECURITY.md) - Security policy

---

## Author

**Varun Pratap Bhardwaj**
*Solution Architect*

SuperLocalMemory V3 - Intelligent local memory system for AI coding assistants.

---

## Questions?

Don't hesitate to ask! We're here to help.

- Open an issue
- Start a discussion
- Reach out to maintainers

**Thank you for contributing to SuperLocalMemory V3!**

Your contributions help make intelligent, local-first memory accessible to everyone.
