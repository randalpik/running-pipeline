"""Client for the unofficial Coros Training Hub API.

This is the same web API ``traininghub.coros.com`` uses, reverse-engineered
(reference: github.com/NYT87/coros-connect). It is undocumented and can change
without notice. Auth is email + md5(password); the returned access token goes
in an ``accessToken`` header. Tokens are single-session — logging in here logs
the account out of the Coros web app, and vice-versa — so we cache the token
and only re-login on an auth failure.

Responses are gzip-encoded and wrap payloads as ``{"result": "0000", "data":
{...}}`` (result "0000" == success).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

API_URLS = {
    "us": "https://teamapi.coros.com",
    "eu": "https://teameuapi.coros.com",
}
SUCCESS = "0000"
USER_AGENT = "max-running-pipeline/1.0"


class CorosAuthError(RuntimeError):
    """Raised when login fails or a token can't be refreshed."""


class CorosApiError(RuntimeError):
    """Raised when the API returns a non-success result that isn't auth."""


class CorosClient:
    def __init__(self, email: str, password: str, *, region: str = "us",
                 token_cache: Path | str | None = None):
        self.email = email
        self.password = password
        self.base = API_URLS[region]
        self._token: str | None = None
        self._user_id: str | None = None
        self._token_cache = Path(token_cache) if token_cache else None
        self._load_cached_token()

    # ---- token persistence ----
    def _load_cached_token(self) -> None:
        if self._token_cache and self._token_cache.exists():
            try:
                data = json.loads(self._token_cache.read_text())
                self._token = data.get("accessToken")
                self._user_id = data.get("userId")
            except (ValueError, OSError):
                pass

    def _save_cached_token(self) -> None:
        if self._token_cache:
            self._token_cache.parent.mkdir(parents=True, exist_ok=True)
            self._token_cache.write_text(
                json.dumps({"accessToken": self._token, "userId": self._user_id}))

    @property
    def user_id(self) -> str | None:
        return self._user_id

    # ---- low-level request ----
    def _request(self, path, *, method="GET", params=None, json_body=None,
                 authed=False):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        body = None
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        if authed:
            if not self._token:
                raise CorosAuthError("not logged in")
            headers["accessToken"] = self._token
            if self._user_id:
                headers["yfheader"] = json.dumps({"userId": self._user_id})
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode())

    @staticmethod
    def _is_auth_failure(resp) -> bool:
        # Coros signals an expired/invalid token with a non-"0000" result;
        # message text varies, so treat any non-success on an authed call as a
        # candidate for one re-login attempt.
        return resp.get("result") not in (SUCCESS, None)

    def _authed(self, path, *, method="GET", params=None, json_body=None):
        """Authed call with one transparent re-login on auth failure."""
        if not self._token:
            self.login()
        resp = self._request(path, method=method, params=params,
                             json_body=json_body, authed=True)
        if self._is_auth_failure(resp):
            self.login()
            resp = self._request(path, method=method, params=params,
                                 json_body=json_body, authed=True)
        if resp.get("result") != SUCCESS:
            raise CorosApiError(f"{path}: {resp.get('result')} {resp.get('message')}")
        return resp["data"]

    # ---- endpoints ----
    def login(self):
        # Announce before the request: login is a throttle-prone call, and a
        # silent stall here is indistinguishable from a hang in CI logs.
        print("[coros] logging in (password auth)…", flush=True)
        resp = self._request("/account/login", method="POST", json_body={
            "account": self.email,
            "accountType": 2,
            "pwd": hashlib.md5(self.password.encode()).hexdigest(),
        })
        if resp.get("result") != SUCCESS or "data" not in resp:
            raise CorosAuthError(f"login failed: {resp.get('result')} "
                                 f"{resp.get('message')}")
        self._token = resp["data"]["accessToken"]
        self._user_id = resp["data"].get("userId")
        self._save_cached_token()
        return self._user_id

    def list_activities(self, *, from_day=None, to_day=None, size=20, page=1):
        """One page of the activity list. from_day/to_day are 'YYYYMMDD'."""
        params = {"size": size, "pageNumber": page}
        if from_day:
            params["startDay"] = from_day
        if to_day:
            params["endDay"] = to_day
        return self._authed("/activity/query", params=params)

    def iter_activities(self, *, from_day=None, to_day=None, page_size=50):
        """Yield every activity in the range, paging until exhausted."""
        page = 1
        while True:
            data = self.list_activities(from_day=from_day, to_day=to_day,
                                        size=page_size, page=page)
            items = data.get("dataList") or []
            total_pages = data.get("totalPage") or 1
            if total_pages > 1:   # liveness on long (cold-cache) listings;
                print(f"[coros] activity list page {page}/{total_pages} "
                      f"({len(items)} items)", flush=True)   # 1-page runs stay quiet
            for it in items:
                yield it
            if page >= total_pages or not items:
                break
            page += 1

    def activity_detail(self, label_id, sport_type):
        return self._authed("/activity/detail/query", method="POST",
                            params={"labelId": label_id, "sportType": sport_type})
