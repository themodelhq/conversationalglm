# Deployment Guide

## Docker Compose

```bash
export POSTGRES_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export GLM_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export GLM_MODEL_DIR="$PWD/checkpoints/glm-sft"
docker compose up --build
```

The API is exposed at port 8000 and Prometheus at port 9090. Use a TLS reverse proxy in front of the API and replace local development origins with your HTTPS frontend origins.

## Kubernetes

Create the secret from an audited secret manager, provision a read-only model volume, then apply the kustomization.

```bash
kubectl apply -f deployment/k8s/namespace.yaml
kubectl -n conversational-glm create secret generic glm-secrets --from-literal=GLM_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
kubectl apply -k deployment/k8s
```

The deployment requests one NVIDIA GPU and uses node label `accelerator=nvidia`. Configure NVIDIA device plugins, metrics-server, a PostgreSQL service, cert-manager, and an ingress controller in the cluster before applying it.

## Operations

Monitor `/health`, `/metrics`, GPU memory, request latency, 429 rate-limit responses, authentication failures, database connection saturation, retrieval hit quality, and model error rate. Pin image digests and scan dependencies before promotion.
