#!/usr/bin/env python3
"""Check shared dependencies only; the retired scaffold has no native trainer."""
import json
import sys
from importlib.metadata import distribution, version


CORE_COMMIT = "b1f780decc516808cfd3019d5e5b674457161f86"


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("The current shared core requires Python >=3.11.")

    core = distribution("particle-sliders-core")
    installed = {"particle-sliders-core": core.version, "particlegan": version("particlegan")}
    expected = {"particle-sliders-core": "0.15.2", "particlegan": "0.11.1"}
    if installed != expected:
        raise SystemExit(f"Install requirements-core.txt in a fresh environment: {installed}")
    source = json.loads(core.read_text("direct_url.json") or "{}")
    if (source.get("url") != "https://github.com/HyperGAN/particle-sliders.git"
            or source.get("subdirectory") != "packages/particle-sliders-core"
            or source.get("vcs_info", {}).get("commit_id") != CORE_COMMIT):
        raise SystemExit(f"Install the pinned shared source from requirements-core.txt ({CORE_COMMIT}).")

    from particle_sliders.adaptive_steering import ActivationGate, CLABSParameters, calibrate_gate_budget
    from particle_sliders.engine import ConditionalParticleGAN, validate_engine_domain
    from particle_sliders.prompt_steering import PromptSteering, psr_mse_loss
    from particle_sliders.session_cli import main as shared_cli

    # The public gate checks the installed PG source manifest without a model,
    # optimizer, training session, GPU allocation or network access.
    profile = validate_engine_domain([1, 4, 8, 8], response_domain="continuous_field")
    entry_points = (ConditionalParticleGAN, shared_cli, ActivationGate, CLABSParameters,
                    calibrate_gate_budget, PromptSteering, psr_mse_loss)
    if not all(callable(entry_point) for entry_point in entry_points):
        raise SystemExit("A shared engine, CLI or steering entry point is unavailable.")
    print(json.dumps({"scope": "shared core only; no native SD trainer",
                      "versions": installed, "core_commit": CORE_COMMIT,
                      "steering_apis": ["CFP", "CLABS", "PSR"], **profile}, sort_keys=True))


if __name__ == "__main__":
    main()
