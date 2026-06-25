from __future__ import annotations

import unittest

from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.local_server_security import (
    SESSION_TOKEN_HEADER,
    validate_local_request,
    validate_loopback_bind_host,
)


class LocalServerSecurityTests(unittest.TestCase):
    def test_bind_host_requires_loopback_without_explicit_unsafe_opt_in(self) -> None:
        validate_loopback_bind_host(
            "127.0.0.1",
            allow_unsafe_host=False,
            server_name="GUI",
        )
        validate_loopback_bind_host(
            "::1",
            allow_unsafe_host=False,
            server_name="GUI",
        )

        with self.assertRaisesRegex(BooguTurboMlxError, "non-loopback"):
            validate_loopback_bind_host(
                "0.0.0.0",
                allow_unsafe_host=False,
                server_name="GUI",
            )

        validate_loopback_bind_host(
            "0.0.0.0",
            allow_unsafe_host=True,
            server_name="GUI",
        )

    def test_local_request_requires_token_and_same_origin_for_mutations(self) -> None:
        token = "session-token"
        headers = {
            "Host": "127.0.0.1:49152",
            "Origin": "http://127.0.0.1:49152",
            SESSION_TOKEN_HEADER: token,
        }

        validate_local_request(
            headers=headers,
            path="/api/generate",
            expected_token=token,
            allow_unsafe_host=False,
            require_same_origin=True,
        )

        with self.assertRaisesRegex(BooguTurboMlxError, "session token"):
            validate_local_request(
                headers={key: value for key, value in headers.items() if key != SESSION_TOKEN_HEADER},
                path="/api/generate",
                expected_token=token,
                allow_unsafe_host=False,
                require_same_origin=True,
            )

        with self.assertRaisesRegex(BooguTurboMlxError, "Origin"):
            validate_local_request(
                headers={**headers, "Origin": "https://example.invalid"},
                path="/api/generate",
                expected_token=token,
                allow_unsafe_host=False,
                require_same_origin=True,
            )

        with self.assertRaisesRegex(BooguTurboMlxError, "Host"):
            validate_local_request(
                headers={**headers, "Host": "192.168.1.2:49152"},
                path="/api/generate",
                expected_token=token,
                allow_unsafe_host=False,
                require_same_origin=True,
            )


if __name__ == "__main__":
    unittest.main()
