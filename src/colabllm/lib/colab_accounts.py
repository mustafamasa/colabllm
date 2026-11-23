"""Read-only helpers for colab-ext multi-account and shared-Drive state.

This module uses only the Python standard library and never imports ``colab_ext``
or reads OAuth / DriveFS credential secrets. When ``accounts.json`` does not exist,
every helper falls back to legacy single-account ``~/.config/colab-cli`` paths.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re

from cli_support import CliError, UsageError

ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,63}$")
_UNSET = object()


def colab_config_dir() -> Path:
    raw = os.environ.get("COLAB_CLI_CONFIG_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".config" / "colab-cli"


def validate_account(value: str) -> str:
    val = value.strip() if isinstance(value, str) else ""
    if not val or ".." in val or not ACCOUNT_RE.fullmatch(val):
        raise UsageError("Account identifiers must be 1-64 letters, digits, '.', '_', '@' or '-', start with a letter or digit, and not contain '..'")
    return val


def load_accounts_registry() -> dict | None:
    path = colab_config_dir() / "accounts.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(f"Colab account configuration is corrupt: {path}; run colab account status to inspect or repair it") from error
    if not isinstance(data, dict) or not isinstance(data.get("accounts", {}), dict):
        raise CliError(f"Invalid Colab account configuration: {path}; accounts must be an object")
    return data


def resolve_account_identifier(identifier: str, registry: dict | None = None) -> str:
    ident = validate_account(identifier)
    if registry is None:
        return ident
    accounts = registry.get("accounts") or {}
    if ident in accounts:
        return ident
    target_lower = ident.lower()
    matches = [
        name
        for name, rec in accounts.items()
        if isinstance(rec, dict) and str(rec.get("email") or "").strip().lower() == target_lower
    ]
    if len(matches) == 1:
        return validate_account(matches[0])
    if len(matches) > 1:
        raise UsageError(f"Email '{ident}' matches multiple Colab accounts ({', '.join(sorted(matches))}); specify an account name")
    available = ", ".join(sorted(accounts)) or "(none)"
    raise UsageError(
        f"Colab account '{ident}' not found; available accounts: {available} (use colab account list or colab account add)"
    )


def _has_multi_account_layout(cfg: Path, account: str | None, registry: dict | None) -> bool:
    if registry is not _UNSET and registry is not None:
        return True
    if (cfg / "accounts.json").is_file():
        return True
    if account and (cfg / "accounts" / account).is_dir():
        return True
    return False


def account_sessions_path(account: str | None = None, *, registry: dict | None = _UNSET) -> Path:
    cfg = colab_config_dir()
    if account and _has_multi_account_layout(cfg, account, registry):
        return cfg / "accounts" / account / "sessions.json"
    return cfg / "sessions.json"


def session_recorded_in_account(
    session_name: str,
    account: str | None,
    *,
    registry: dict | None = _UNSET,
) -> bool:
    path = account_sessions_path(account, registry=registry)
    if not path.is_file():
        return False
    with contextlib.suppress(OSError, ValueError, UnicodeError):
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and session_name in data
    return False


def resolve_effective_account(
    *,
    explicit: str | None = None,
    saved: str | None = None,
    prefer_saved: bool = False,
    reattach_session: str | None = None,
    registry: dict | None = _UNSET,
) -> tuple[str | None, str]:
    reg = load_accounts_registry() if registry is _UNSET else registry
    if explicit:
        return resolve_account_identifier(explicit, reg), "flag"

    env_val = os.environ.get("COLABLLM_ACCOUNT", "").strip()
    if env_val:
        return resolve_account_identifier(env_val, reg), "env"

    if prefer_saved and saved:
        return validate_account(saved), "state"

    colab_env_val = os.environ.get("COLAB_ACCOUNT", "").strip()
    if colab_env_val:
        return resolve_account_identifier(colab_env_val, reg), "colab_env"

    if not prefer_saved and saved and reattach_session:
        valid_saved = validate_account(saved)
        accounts = (reg.get("accounts") or {}) if reg is not None else {}
        if (reg is None or valid_saved in accounts) and session_recorded_in_account(
            reattach_session, valid_saved, registry=reg
        ):
            return valid_saved, "reattach"

    if reg is not None:
        accounts = reg.get("accounts") or {}
        active = str(reg.get("active") or "").strip()
        if active and (active in accounts or not accounts):
            return validate_account(active), "active"

    return None, "none"


def resolve_effective_drive_account(
    colab_account: str | None,
    *,
    explicit_drive: str | None = None,
    saved_drive: str | None = None,
    prefer_saved: bool = False,
    registry: dict | None = _UNSET,
) -> tuple[str | None, str]:
    reg = load_accounts_registry() if registry is _UNSET else registry
    if explicit_drive:
        return resolve_account_identifier(explicit_drive, reg), "flag"

    env_val = os.environ.get("COLABLLM_DRIVE_ACCOUNT", "").strip()
    if env_val:
        return resolve_account_identifier(env_val, reg), "env"

    if prefer_saved and saved_drive:
        return validate_account(saved_drive), "state"

    colab_drv = (
        os.environ.get("COLAB_DRIVE_ACCOUNT")
        or os.environ.get("COLAB_DFS_ACCOUNT")
        or ""
    ).strip()
    if colab_drv:
        return resolve_account_identifier(colab_drv, reg), "colab_env"

    if saved_drive:
        return validate_account(saved_drive), "state"

    if reg is not None and colab_account:
        cfg = colab_config_dir()
        accounts = reg.get("accounts") or {}
        rec = accounts.get(colab_account) if isinstance(accounts.get(colab_account), dict) else {}
        bound = str((rec or {}).get("drive_account") or "").strip()
        if bound and bound in accounts:
            return validate_account(bound), "bound"
        if (cfg / "accounts" / colab_account / "dfs_credentials.json").is_file():
            return colab_account, "self"
        default_drv = str(reg.get("default_drive_account") or "").strip()
        if default_drv and default_drv in accounts:
            return validate_account(default_drv), "default"
        return colab_account, "self"

    return None, "none"


def account_dfs_credentials_path(
    colab_account: str | None = None,
    drive_account: str | None = None,
    *,
    registry: dict | None = _UNSET,
) -> Path:
    cfg = colab_config_dir()
    target = drive_account or colab_account
    if target and _has_multi_account_layout(cfg, target, registry):
        return cfg / "accounts" / target / "dfs_credentials.json"
    return cfg / "dfs_credentials.json"


def colab_subprocess_env(
    account: str | None = None,
    drive_account: str | None = None,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    src = os.environ if base_env is None else base_env
    if (
        account is None
        and drive_account is None
        and not any(k in src for k in ("COLAB_ACCOUNT", "COLAB_DRIVE_ACCOUNT", "COLAB_DFS_ACCOUNT"))
    ):
        return None
    env = dict(src)
    if account:
        env["COLAB_ACCOUNT"] = account
    if drive_account:
        env["COLAB_DRIVE_ACCOUNT"] = drive_account
        env.pop("COLAB_DFS_ACCOUNT", None)
    elif account:
        env.pop("COLAB_DRIVE_ACCOUNT", None)
        env.pop("COLAB_DFS_ACCOUNT", None)
    return env


def list_account_names() -> list[str]:
    """Return candidate Colab account names offline without reading credentials or keyring."""
    cfg = colab_config_dir()
    names: set[str] = set()
    accounts_dir = cfg / "accounts"
    if accounts_dir.is_dir():
        with contextlib.suppress(OSError):
            for entry in accounts_dir.iterdir():
                if entry.is_dir() and ".." not in entry.name and ACCOUNT_RE.fullmatch(entry.name):
                    names.add(entry.name)
    reg_path = cfg / "accounts.json"
    if reg_path.is_file():
        with contextlib.suppress(OSError, ValueError, UnicodeError):
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("accounts"), dict):
                for key in data["accounts"]:
                    if isinstance(key, str) and ".." not in key and ACCOUNT_RE.fullmatch(key):
                        names.add(key)
    return sorted(names)
