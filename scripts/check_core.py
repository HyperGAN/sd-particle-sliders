#!/usr/bin/env python3
"""Check shared dependencies only; the retired scaffold has no native trainer."""
import json
import sys
from importlib.metadata import version


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("The current shared core requires Python >=3.11.")

    installed = {name: version(name) for name in ("particle-sliders-core", "particlegan")}
    expected = {"particle-sliders-core": "0.12.0", "particlegan": "0.9.4"}
    if installed != expected:
        raise SystemExit(f"Install requirements-core.txt in a fresh environment: {installed}")

    from particle_sliders.pg094 import ConditionalPG094, validate_pg09_domain
    from particle_sliders.session_cli import main as shared_cli

    # The public gate checks the installed PG source manifest without a model,
    # optimizer, training session, GPU allocation or network access.
    profile = validate_pg09_domain([1, 4, 8, 8], response_domain="continuous_field")
    if not callable(ConditionalPG094) or not callable(shared_cli):
        raise SystemExit("The shared engine or CLI entry point is unavailable.")
    print(json.dumps({"scope": "shared core only; no native SD trainer",
                      "versions": installed, **profile}, sort_keys=True))


if __name__ == "__main__":
    main()
