# Enterprise Deployment Guide

This guide covers deploying MechanicsDSL in enterprise environments.

## Docker Deployment

### Quick Start

```bash
# Build image
docker build -t mechanicsdsl:2.0 .

# Run API server
docker run -d -p 8000:8000 mechanicsdsl:2.0 \
    python -m mechanics_dsl.server
```

### Docker Compose (Recommended)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Scale workers
docker compose up -d --scale worker=4
```

### GPU Support

```bash
# Build GPU image
docker build --target runtime-gpu -t mechanicsdsl:2.0-gpu .

# Run with GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Kubernetes Deployment

### Helm Chart (Coming Soon)

```bash
helm repo add mechanicsdsl https://charts.mechanicsdsl.io
helm install mdsl mechanicsdsl/mechanicsdsl
```

### Manual Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mechanicsdsl-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mechanicsdsl
  template:
    metadata:
      labels:
        app: mechanicsdsl
    spec:
      containers:
      - name: api
        image: mechanicsdsl:2.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

## Internal PyPI Mirror

For air-gapped environments, mirror MechanicsDSL using bandersnatch or Nexus:

### Nexus Configuration

1. Create PyPI proxy repository
2. Add to pip.conf:

```ini
[global]
index-url = https://nexus.internal.company/repository/pypi-proxy/simple
trusted-host = nexus.internal.company
```

### bandersnatch

```yaml
[mirror]
directory = /srv/pypi
json = true
master = https://pypi.org

[plugins]
enabled = allowlist_project

[allowlist]
packages = 
    mechanicsdsl-core
    numpy
    scipy
    sympy
    matplotlib
```

## Security Considerations

### Sandboxing Simulations

```python
from mechanics_dsl import PhysicsCompiler

# Restrict file system access
compiler = PhysicsCompiler(
    sandbox=True,  # Prevents file I/O in DSL code
    max_memory="2GB",
    max_time=300  # 5 minute timeout
)
```

### Input Validation

The DSL parser validates input to prevent code injection:

```python
# Safe - DSL syntax only
compiler.compile_dsl(r"\system{pendulum} ...")

# Blocked - Python code rejected
compiler.compile_dsl("import os; os.system('rm -rf /')")
# Raises ParseError
```

## Monitoring

### Prometheus Metrics

```python
# Enable metrics endpoint
python -m mechanics_dsl.server --metrics-port 9090
```

Available metrics:
- `mdsl_simulations_total` - Total simulations run
- `mdsl_simulation_duration_seconds` - Simulation time histogram
- `mdsl_active_connections` - WebSocket connections

### Logging

```python
import logging
logging.getLogger('mechanics_dsl').setLevel(logging.INFO)
```

## Performance Tuning

### Recommended Settings

| Environment | Workers | Memory | Notes |
|-------------|---------|--------|-------|
| Development | 1 | 1 GB | Single thread |
| Production | 4 | 4 GB | Per CPU core |
| GPU | 1 | 8 GB | CUDA memory |

### Environment Variables

```bash
export MDSL_LOG_LEVEL=INFO
export MDSL_MAX_WORKERS=4
export MDSL_CACHE_DIR=/tmp/mdsl_cache
export OMP_NUM_THREADS=4
```

## Support

- GitHub Issues: https://github.com/MechanicsDSL/mechanicsdsl/issues
- Documentation: https://mechanicsdsl.readthedocs.io
- Email: support@mechanicsdsl.io
