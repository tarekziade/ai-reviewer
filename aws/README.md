# AWS deployment

This directory provisions a single Amazon Linux 2023 EC2 instance for the
interactive reviewboard app (`reviewbot-web`).

`deploy.sh` will:

- launch an EC2 instance in the selected subnet/VPC
- clone this repo into `/opt/app/ai-reviewer`
- install `.[web]` into a local virtualenv
- copy `reviewbot-web.env` to `/etc/reviewbot/reviewbot-web.env`
- install and enable `reviewbot-web.service`
- start the app on port `8080`

## Usage

1. Copy `reviewbot-web.env.example` to `reviewbot-web.env`.
2. Fill in the GitHub App, GitHub OAuth, and LLM credentials.
3. If you use `GITHUB_PRIVATE_KEY_PATH=/path/to/key.pem` like your local
   launch command, keep that local path in `reviewbot-web.env`.
   `deploy.sh` will copy the PEM to `/etc/reviewbot/github-app.pem` on
   the instance and rewrite the env file to point there.
4. Run `./aws/deploy.sh`.

The instance runs without a public IP. Use the printed private IP over your
private network/VPN, then open `http://<private-ip>:8080`.

If you want the same no-auth behavior as your local command, keep
`DEV_NO_AUTH=1` in `reviewbot-web.env`. On a shared or public host,
remove that and configure `WEB_ALLOWED_USERS` or `WEB_ALLOWED_ORG`.

## Updating an existing deployment

`./aws/update.sh` refreshes an already-deployed box in place — no
instance churn, no re-key, no downtime beyond a `systemctl restart`. It:

- reads `.deploy-state.json` to find the instance + key
- refreshes the cached private IP if AWS moved it
- rsyncs the local working tree to `/opt/app/ai-reviewer` (with
  `--delete`, excluding `.git`, `.venv`, `aws/`, caches, and macOS
  noise — so PEMs and local state never leave your machine)
- `pip install -e '.[web]'` to pick up any new dependencies
- rewrites `/etc/reviewbot/${SERVICE_NAME}.env` (and the PEM if your
  `GITHUB_PRIVATE_KEY_PATH` still points at a local file)
- `sudo systemctl restart` the service and prints its status

Use it whenever you've changed code locally or edited
`reviewbot-web.env`. Because it rsyncs your working tree, what's on
the host matches what's on your laptop right now — uncommitted
changes included. If you'd rather ship only what's pushed, commit
+ check out the deployment branch before running.

Destroy the stack with `./aws/destroy.sh`.

## Security notes

- The instance is launched without a public IP and listens on
  `0.0.0.0:8080` plain HTTP. That's safe **only** while the box sits
  inside a VPN / private network. If you ever expose it to the public
  internet, terminate TLS in front of it (e.g. nginx + Let's Encrypt)
  and flip `DEV_NO_AUTH` off — the session cookie's `Secure` flag is
  tied to `DEV_NO_AUTH=1`, so plain HTTP works only in that mode.
- `WEB_SESSION_SECRET` must be a real random value. `deploy.sh` and
  `update.sh` mint one with `openssl rand -hex 32` if your env still has
  the example placeholder, and write the value back to
  `reviewbot-web.env` so subsequent updates reuse it.
- The PEM and env files land on the host as `0600` owned by
  `ec2-user` — only the service account can read the GitHub App private
  key, LLM API key, OAuth client secret, and session secret.
- `ALLOW_APPROVE` defaults to off. The web UI relies on this to refuse
  publishing an LLM-chosen `APPROVE` event, because that event is
  influenced by attacker-controlled PR content. Turn it on only after
  deciding your operators will verify every APPROVE before clicking
  publish.
