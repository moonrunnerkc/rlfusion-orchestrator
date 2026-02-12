# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

RLFusion Orchestrator includes AI safety layers, RL-based routing, and processes user queries locally. Security issues in any of these components could have serious consequences.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **Contact:** Reach out directly to **Bradley R. Kinnard** via [LinkedIn](https://www.linkedin.com/in/brad-kinnard/) with the subject line **"RLFusion Security Disclosure"**.
2. **Include:**
   - A clear description of the vulnerability.
   - Steps to reproduce the issue.
   - The potential impact (e.g., data exposure, prompt injection bypass, policy manipulation).
   - Any suggested fix, if you have one.
3. **Response time:** You will receive an acknowledgment within **48 hours** and a detailed response within **7 business days**.

### What Qualifies

- Prompt injection that bypasses the safety/critique layers.
- Unauthorized access to the API without rate limiting.
- RL policy manipulation or poisoning vectors.
- Information leakage through retrieval responses.
- Dependency vulnerabilities with a known exploit path.
- WebSocket endpoint abuse or denial of service vectors.

### What Does Not Qualify

- Issues requiring physical access to the machine.
- Vulnerabilities in upstream dependencies with no exploit path in this project.
- Feature requests or usability concerns (use GitHub Issues for those).

### Disclosure Policy

- We follow coordinated disclosure. We will work with you on a timeline for the fix.
- Credit will be given in the release notes unless you prefer to remain anonymous.

## Security Best Practices for Users

- **Never commit `.env` files** — they contain API keys. The `.gitignore` already excludes them.
- **Run behind a reverse proxy** (e.g., nginx) if exposing the API beyond localhost.
- **Keep dependencies updated** — run `pip install --upgrade -r backend/requirements.txt` periodically.
- **Review `config.yaml`** before deployment — ensure web search is disabled unless intentionally needed.
