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

Destroy the stack with `./aws/destroy.sh`.
