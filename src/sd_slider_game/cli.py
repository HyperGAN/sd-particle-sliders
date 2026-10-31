"""Explicit opt-in CLI; native data are local files, never implicit downloads."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-game', action='store_true', help='accepted by historical train_lora dispatch')
    parser.add_argument('--config', type=Path, help='JSON mapping of explicit modern session options')
    parser.add_argument('--fixture', action='store_true', help='tiny random actual Diffusers CPU API fixture')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    if bool(args.config) == args.fixture:
        parser.error('choose --config or --fixture')
    from particle_sliders.response import open_response
    from particle_sliders.inference import export_inference
    from .adapter import config
    options = json.loads(args.config.read_text()) if args.config else dict(fixture=True, ema_decay=.9)
    with open_response(config(**options), resume=args.resume) as session:
        session.train(stop_after=args.stop_after)
        args.output.mkdir(parents=True, exist_ok=True)
        session.checkpoint(args.output / 'training.sd-pg094.pt')
        export_inference(session, args.output / 'adapter.sd-pg094.raw.pt', selection='raw')
        if session.ema.values is not None:
            export_inference(session, args.output / 'adapter.sd-pg094.ema.pt', selection='ema')
        print(json.dumps(dict(completed_steps=session.game.completed_steps, cohort=session.config.cohort,
                              response_shape=session.game.response_shape, qualification='CPU API fixture only'
                              if args.fixture else 'staged model; pretrained quality unqualified')))


if __name__ == '__main__':
    main()
