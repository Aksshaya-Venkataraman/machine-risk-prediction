# Security

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's **private vulnerability reporting**
(Security → Report a vulnerability) on this repository.

## Controls in place

| Concern | Control |
|---|---|
| Secrets | Read only from environment variables; `.env` is git-ignored and created with mode 600; services refuse weak or placeholder secrets; gitleaks scans the full git history in CI |
| Authentication | Argon2id password hashing (automatic rehash on login), constant-time login path for unknown emails, JWT HS256 with pinned algorithm, issuer and expiry checks, 30-minute tokens |
| Authorisation | Users can only read or delete their own predictions (ownership is part of every query, so other users' IDs return 404); admin routes require the admin role, checked from the database on every request; admins cannot demote themselves |
| Service-to-service | Model service requires a bearer token (constant-time comparison); reload uses a separate admin token; the model service is on an internal network only |
| Input validation | Strict Pydantic schemas (`extra="forbid"`, strict types, regex allow-lists for IDs and feature names, size limits); request bodies over the limit get 413 before parsing |
| XSS / injection | JSON-only API; `Content-Security-Policy: default-src 'none'`; SQLAlchemy ORM with bound parameters only |
| Abuse | Rate limits per IP (login, register) and per user (API), shared through Redis |
| Transport | Caddy terminates TLS with automatic certificates; HSTS; only ports 80/443 published |
| Headers | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, `Cache-Control: no-store`, COOP |
| CORS | Explicit allow-list from `CORS_ORIGINS`; wildcards rejected; no credentials |
| Production mode | `debug=False`, OpenAPI/Swagger disabled, generic 500 responses (no stack traces), trusted-host check |
| Containers | Non-root users, read-only root filesystems, `no-new-privileges`, all Linux capabilities dropped |
| Database | App connects as a non-superuser role that owns only its own database; PostgreSQL port not published; SCRAM-SHA-256 auth |
| Model files | Loaded only if their SHA-256 matches the training manifest (joblib uses pickle) |
| Dependencies | Exact versions locked with hashes (`pip install --require-hashes`), pip-audit and Bandit in CI, Dependabot weekly |

## Operator checklist

- Generate secrets with `python3 scripts/generate_env.py`. Never reuse them across environments.
- Rotate `JWT_SECRET_KEY` to sign every user out; rotate the internal tokens on both services together.
- Back up the `db_data` volume. It holds user accounts and prediction history.
