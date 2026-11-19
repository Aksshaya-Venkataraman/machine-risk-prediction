"""Operator commands.

    python -m gateway.cli create-admin --email ops@example.com

The password is read from ADMIN_PASSWORD or prompted for; it is never passed
on the command line, where it would end up in shell history.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from pydantic import ValidationError
from sqlalchemy import select

from gateway.config import get_settings
from gateway.db import User, make_engine, make_session_factory
from gateway.schemas import RegisterRequest
from gateway.security import hash_password


def create_admin(email: str, password: str) -> str:
    try:
        data = RegisterRequest(email=email, password=password)
    except ValidationError as exc:
        raise SystemExit(f"invalid email or password: {exc.errors()[0]['msg']}") from None
    engine = make_engine(get_settings().database_url.get_secret_value())
    with make_session_factory(engine)() as db:
        user = db.scalar(select(User).where(User.email == data.email))
        if user is None:
            user = User(email=data.email, password_hash=hash_password(data.password), role="admin")
            db.add(user)
            action = "created"
        else:
            user.role, user.is_active = "admin", True
            user.password_hash = hash_password(data.password)
            action = "promoted"
        db.commit()
    engine.dispose()
    return action


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gateway.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    admin = sub.add_parser("create-admin", help="create or promote an admin user")
    admin.add_argument("--email", required=True)
    args = parser.parse_args(argv)

    password = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("Admin password (min 12 chars): ")
    action = create_admin(args.email, password)
    print(f"admin {args.email} {action}", file=sys.stderr)


if __name__ == "__main__":
    main()
