# Security Policy

## Reporting a Vulnerability

MechanicsDSL includes an exec()-free compiler architecture designed to prevent arbitrary code execution from DSL files. However, if you discover a security vulnerability, please report it responsibly.

- Do **not** open a public issue.
- Email the project maintainer directly at [mechanicsdsl.project@gmail.com](mailto:mechanicsdsl.project@gmail.com).
- We will acknowledge your report within 48 hours and provide a timeline for a patch.

---

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.0.x   | :white_check_mark: |
| 1.5.x   | :white_check_mark: |
| < 1.5   | :x:                |

---

## Security Features

### Input Validation

The `security.py` module provides centralized security validation:

```python
from mechanics_dsl.security import validate_dsl_code, InjectionError

# Validates DSL code and blocks dangerous patterns
try:
    safe_code = validate_dsl_code(user_input)
except InjectionError as e:
    print(f"Blocked: {e}")
```

**Blocked patterns:**
- `eval()`, `exec()`, `compile()`
- `__import__`, `importlib`
- `os.system`, `subprocess`, `popen`
- `pickle.load`, `marshal.load`
- `open()` with dangerous modes
- `shell=True` arguments

### Path Traversal Protection

```python
from mechanics_dsl.security import validate_path, PathTraversalError

# Validates paths and prevents directory escape
safe_path = validate_path(user_path, base_dir="/safe/directory")
```

**Blocked:**
- `..` traversal sequences
- Null byte injection
- Symlink escapes
- Absolute path breakouts

### Sandbox Execution

```python
from mechanics_dsl.security import Sandbox, SandboxConfig

# Execute untrusted code with resource limits
config = SandboxConfig(max_time_seconds=60, max_memory_mb=512)
with Sandbox(config) as sb:
    result = sb.execute(run_simulation, args=(data,))
```

### Rate Limiting

```python
from mechanics_dsl.utils.rate_limit import RateLimiter

limiter = RateLimiter(requests_per_minute=60)
if not limiter.allow(client_id):
    raise TooManyRequestsError()
```

---

## Continuous Security Scanning

Our CI/CD pipeline includes:

| Tool | Purpose |
|------|---------|
| **CodeQL** | Static analysis for Python vulnerabilities |
| **Bandit** | Python security linter |
| **Semgrep** | Pattern-based security scanning |
| **Trivy** | Container and dependency scanning |
| **Gitleaks** | Secret detection |
| **Safety** | Dependency vulnerability checks |

See `.github/workflows/security.yml` for details.

---

## Best Practices for Users

1. **Never use `pickle` for untrusted data**
   - Pickle can execute arbitrary code
   - Use JSON for data serialization when possible

2. **Validate all user inputs**
   ```python
   from mechanics_dsl.security import validate_identifier, validate_number
   
   name = validate_identifier(user_name)
   value = validate_number(user_value, min_val=0, max_val=1000)
   ```

3. **Use sandboxed execution for untrusted DSL**
   ```python
   compiler = PhysicsCompiler()
   result = compiler.compile_dsl(untrusted_code)  # Automatically validated
   ```

4. **Keep dependencies updated**
   - Run `pip install --upgrade mechanics-dsl` regularly
   - Review security advisories

---

## Known Security Considerations

| Issue | Mitigation |
|-------|------------|
| `pickle` in exports | Only use trusted `.pkl` files |
| Large simulation DoS | Set `max_time_seconds` limits |
| Memory exhaustion | Configure `max_memory_mb` |
| File system access | Use `validate_path()` with `base_dir` |

---

## Disclosure Timeline

| Date | Event |
|------|-------|
| Day 0 | Vulnerability reported |
| Day 2 | Acknowledgment sent |
| Day 7 | Assessment completed |
| Day 14-30 | Patch developed and tested |
| Day 30 | Coordinated disclosure |

---

## Security Contacts

- **Primary**: mechanicsdsl.project@gmail.com
- **Backup**: Open a private security advisory on GitHub
