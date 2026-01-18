# MechanicsDSL Kubernetes Deployment

Kubernetes manifests for deploying MechanicsDSL as a service.

## Components

| File | Description |
|------|-------------|
| `deployment.yaml` | Main application deployment |
| `service.yaml` | Load balancer service |
| `configmap.yaml` | Configuration settings |
| `hpa.yaml` | Horizontal Pod Autoscaler |

## Quick Start

```bash
# Create namespace
kubectl create namespace mechanicsdsl

# Apply all manifests
kubectl apply -f kubernetes/ -n mechanicsdsl

# Check status
kubectl get pods -n mechanicsdsl
```

## Configuration

Edit `configmap.yaml` to customize:
- Number of workers
- Memory limits
- Log levels
- Feature flags

## Scaling

The HPA configuration will automatically scale based on CPU usage:
- Min replicas: 2
- Max replicas: 10
- Target CPU: 70%

## Monitoring

Prometheus metrics available at `/metrics` endpoint.
