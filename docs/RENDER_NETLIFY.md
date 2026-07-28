# Render and Netlify Deployment

## Render API deployment

1. Push this repository to GitHub, GitLab, or Bitbucket.
2. In Render, create a Blueprint from the repository. Render reads `render.yaml` and provisions PostgreSQL, the API service, and a persistent `/var/data` disk.
3. Copy the public Render API URL without a trailing slash, for example `https://conversational-glm-api.onrender.com`.
4. Set `GLM_CORS_ORIGINS` to the Netlify production URL if you enable direct browser-to-Render requests. The default Netlify proxy does not require browser CORS.

`Dockerfile.render` installs the API, training orchestration, and base ML runtime. It intentionally excludes local optional diffusion and Coqui TTS extras to fit a web-service deployment. Deploy dedicated GPU workers with the full `Dockerfile` for heavy image, speech-synthesis, and video workloads.

Leave `GLM_MODEL_PATH` unset until a complete exported model exists on the persistent disk. A valid local model directory must contain `config.json`, `model.safetensors`, and `tokenizer.json`. If the variable points at a missing or incomplete directory, the API starts in platform-only mode and `/health` reports `model_status: unavailable`; training, data governance, and run observability remain available.

## Netlify web deployment

1. In Netlify, add the same repository as a new site.
2. In **Build & deploy → Continuous deployment**, set the Base directory to `frontend`, or leave it blank and let the repository-root `netlify.toml` select `frontend`.
3. Netlify reads `netlify.toml`, installs only frontend Node dependencies, builds the dashboard, and deploys the `glm-api` serverless proxy function.
4. In **Site configuration → Environment variables**, add the following variables:

```env
NODE_VERSION=20
GLM_API_ORIGIN=https://conversational-glm-api.onrender.com
```

`GLM_API_ORIGIN` is private to the Netlify serverless function. Do not name it `VITE_API_URL`; `VITE_*` variables are embedded in browser JavaScript and are public.

5. Clear the Netlify build cache and deploy again after pulling this revision.
6. Open the Netlify URL and create or sign in to an account.

The browser calls same-origin `/api/v1/...`. Netlify rewrites that request to `/.netlify/functions/glm-api`; the function forwards it to `GLM_API_ORIGIN`. This avoids cross-origin browser failures and fixes the login `Failed to fetch` problem caused by missing or mismatched CORS configuration. The proxy returns a descriptive JSON error when `GLM_API_ORIGIN` is missing or Render is unavailable, rather than a generic function failure.

## Optional direct browser API mode

The default proxy mode is recommended. To bypass it, configure these **public** Netlify build variables and redeploy:

```env
VITE_DIRECT_API=true
VITE_API_URL=https://conversational-glm-api.onrender.com
```

When using direct mode, add the exact Netlify production origin to Render:

```env
GLM_CORS_ORIGINS=https://your-site.netlify.app
GLM_CORS_ORIGIN_REGEX=https://.*\.netlify\.app
```

## Persistent state and workers

Render persistent disks retain user uploads, run manifests, logs, tokenizers, and checkpoints under `/var/data`. The Render image initializes the mount before starting Uvicorn, and the application verifies it is writable at startup. Check `/health`: `persistent_storage` must be `true` and `storage_dir` must be `/var/data`. If a disk is absent or mounted read-only, the service falls back to `/tmp/conversational-glm-state` so it can start; that fallback is ephemeral and must not be used for training artifacts. PostgreSQL retains users, conversation metadata, document metadata, dataset registrations, and training-run records.

The web service safely exposes the dashboard and API but is not a replacement for an elastic GPU job scheduler. Use the platform's generated run manifests with a dedicated Render GPU worker, Kubernetes, Slurm, or another authenticated GPU execution service for long-running multi-GPU and multi-node jobs. Keep API replicas at one when using the built-in local process manager, or replace `TrainingManager._launcher` with scheduler submission before horizontal scaling.
