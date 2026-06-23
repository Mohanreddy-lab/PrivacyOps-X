# Security Audit

Date: 2026-04-25

## Scope
- FastAPI/OpenEnv service in `server/`
- Typed input models in `models.py`
- Demo and utility scripts in `scripts/`
- Root Python entrypoints and docs for secret handling

## Findings Fixed
- Added request-body validation and payload-size enforcement in `server/app.py`
- Added per-endpoint IP-based rate limiting in `server/app.py`
- Added model-level input sanitization for `target_id`, `query`, `content`, and string `field_value` in `models.py`
- Added `.env` / key material ignore rules in `.gitignore`
- Added `.env.example` so sensitive runtime config stays out of source control

## Secret Scan Result
- No live hardcoded API keys, bearer tokens, passwords, or private secrets were found in application code
- Existing references to `OPENAI_API_KEY`, `HF_TOKEN`, and `HF_ROUTER_URL` are environment-variable lookups or README examples

## Remaining Risks
- Rate limiting is in-memory and per-process; it is effective for local/demo deployments but not a distributed production-grade control
- There are no auth endpoints in the current benchmark; the stricter `5 attempts / 15 minutes` policy is wired for future auth-like routes automatically, but there is no user authentication system today
- This audit is static and repo-local; it does not include an online CVE/dependency database scan
- The benchmark intentionally exposes public judge-facing endpoints such as `/schema`, `/judge-report`, `/curriculum`, and `/dashboard`

## Recommended Next Steps
- Keep deployment secrets only in Hugging Face Space or local environment settings
- Run an external dependency CVE scan before any broader internet-facing deployment
- If auth is added later, reuse the existing auth-route limiter and add account lockout plus audit logging
