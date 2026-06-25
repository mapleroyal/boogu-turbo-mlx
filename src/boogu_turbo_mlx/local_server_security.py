from __future__ import annotations

import ipaddress
import secrets
from collections.abc import Mapping
from urllib.parse import parse_qs, urlparse

from .errors import BooguTurboMlxError


SESSION_TOKEN_FIELD = "session_token"
SESSION_TOKEN_HEADER = "X-Boogu-Session-Token"
SESSION_TOKEN_QUERY = "token"


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def validate_loopback_bind_host(
    host: str,
    *,
    allow_unsafe_host: bool,
    server_name: str,
) -> None:
    if allow_unsafe_host:
        return
    if not is_loopback_host(host):
        raise BooguTurboMlxError(
            f"Refusing to bind {server_name} to non-loopback host {host!r}. "
            "Pass --unsafe-host only on a trusted network."
        )


def validate_local_request(
    *,
    headers: Mapping[str, str],
    path: str,
    expected_token: str,
    allow_unsafe_host: bool,
    require_same_origin: bool,
) -> None:
    _validate_host_header(headers, allow_unsafe_host=allow_unsafe_host)
    validate_session_token(_request_token(headers, path), expected_token)
    if require_same_origin:
        _validate_same_origin(headers)


def validate_session_token(submitted: str | None, expected: str) -> None:
    if not submitted or not secrets.compare_digest(str(submitted), expected):
        raise BooguTurboMlxError("Invalid or missing session token.")


def token_from_path(path: str) -> str | None:
    values = parse_qs(urlparse(path).query).get(SESSION_TOKEN_QUERY)
    if not values:
        return None
    return values[0]


def is_loopback_host(host: str) -> bool:
    text = _normalize_host(host)
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _request_token(headers: Mapping[str, str], path: str) -> str | None:
    header_token = headers.get(SESSION_TOKEN_HEADER)
    if header_token:
        return str(header_token).strip()
    return token_from_path(path)


def _validate_host_header(
    headers: Mapping[str, str],
    *,
    allow_unsafe_host: bool,
) -> None:
    host = _host_header_hostname(headers.get("Host"))
    if host is None:
        raise BooguTurboMlxError("Invalid or missing Host header.")
    if not allow_unsafe_host and not is_loopback_host(host):
        raise BooguTurboMlxError(
            f"Rejected request with non-loopback Host header {host!r}."
        )


def _validate_same_origin(headers: Mapping[str, str]) -> None:
    request_origin = _origin_from_host_header(headers.get("Host"))
    if request_origin is None:
        raise BooguTurboMlxError("Invalid or missing Host header.")

    for header_name in ("Origin", "Referer"):
        value = headers.get(header_name)
        if not value:
            continue
        origin = _origin_from_url(str(value))
        if origin != request_origin:
            raise BooguTurboMlxError(
                f"Rejected request with invalid {header_name} header."
            )


def _origin_from_host_header(value: str | None) -> tuple[str, str, int] | None:
    parsed = _parse_host_header(value)
    if parsed is None or parsed.hostname is None:
        return None
    try:
        port = parsed.port if parsed.port is not None else 80
    except ValueError:
        return None
    return ("http", _normalize_host(parsed.hostname), int(port))


def _origin_from_url(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return None
        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port if parsed.port is not None else default_port
    except ValueError:
        return None
    return (parsed.scheme, _normalize_host(parsed.hostname), int(port))


def _host_header_hostname(value: str | None) -> str | None:
    parsed = _parse_host_header(value)
    if parsed is None or parsed.hostname is None:
        return None
    return parsed.hostname


def _parse_host_header(value: str | None):
    if not value:
        return None
    try:
        return urlparse(f"http://{value}")
    except ValueError:
        return None


def _normalize_host(host: str) -> str:
    return str(host).strip().strip("[]").lower()
