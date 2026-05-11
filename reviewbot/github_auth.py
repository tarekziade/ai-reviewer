import time

import jwt
import requests


def app_jwt(app_id: str, private_key: str) -> str:
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 9 * 60, "iss": app_id}
    return jwt.encode(payload, private_key, algorithm="RS256")


def installation_token(app_id: str, private_key: str, installation_id: int) -> str:
    token = app_jwt(app_id, private_key)
    r = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["token"]


def installation_id_for_repo(
    app_id: str, private_key: str, owner: str, repo: str
) -> int:
    """Look up the installation id for a repo via the App JWT. Used by
    web mode to mint an installation token without relying on an
    incoming webhook payload to supply the id."""
    token = app_jwt(app_id, private_key)
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/installation",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    iid = data.get("id")
    if not isinstance(iid, int):
        raise RuntimeError(
            f"GitHub returned no installation id for {owner}/{repo}: {data!r}"
        )
    return iid
