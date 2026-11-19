"""Execute the predeclared ablation/seed matrix, reusing only verified completed runs."""
import argparse
from argparse import Namespace
import json
from pathlib import Path

from scripts.train_compact import train, prepare_training_cache, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def verify_frozen_run(directory):
    directory = Path(directory)
    try:
        frozen = json.loads((directory/'frozen-selection.json').read_text(encoding='utf-8'))
        if frozen.get('status') != 'frozen':
            raise ValueError('Run has no frozen selection')
        if frozen.get('optimizer_splits') != ['train'] or frozen.get('selection_split') != 'valid' or frozen.get('test_accessed') is not False:
            raise ValueError('Run has an invalid train/validation-only selection protocol')
        for name, key in [('best.pth', 'checkpoint_sha256'), ('config.json', 'config_sha256'), ('history.json', 'history_sha256')]:
            if sha256(directory/name) != frozen.get(key):
                raise ValueError(f'Frozen run hash mismatch: {name}')
        config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        if config.get('status') != 'frozen' or config.get('data_fingerprint') != frozen.get('cache_fingerprint'):
            raise ValueError('Frozen run configuration/data fingerprint differs')
        return frozen, config
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f'Existing run is not a complete verified frozen experiment: {directory}') from error


def run_suite(args):
    spec = json.loads((ROOT/'config/experiments/compact_ablation.json').read_text(encoding='utf-8'))
    available = {row['name']: row for row in spec['runs']}
    requested = getattr(args, 'cases', None) or list(available)
    if len(set(requested)) != len(requested) or set(requested)-available.keys():
        raise ValueError('Unknown or duplicate experiment cases')
    cache = prepare_training_cache(args.data_dir, args.cache_dir, args.image_size)
    protocol = {'format': spec['format'], 'cases': requested, 'epochs': args.epochs, 'device_request': args.device,
                'image_size': args.image_size, 'base_channels': args.base_channels, 'batch_size': args.batch_size,
                'learning_rate': spec['learning_rate'], 'weight_decay': spec['weight_decay'],
                'data_fingerprint': cache['fingerprint'], 'test_accessed': False,
                'declared_protocol_sha256': sha256(ROOT/'config/experiments/compact_ablation.json')}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Existing suite protocol differs; use a new output directory')
    else:
        write_json(protocol_path, protocol)
    results = []
    for name in requested:
        case, directory = available[name], output/name
        training_args = Namespace(data_dir=Path(args.data_dir), cache_dir=Path(args.cache_dir), output=directory,
                                  image_size=args.image_size, base_channels=args.base_channels, epochs=args.epochs,
                                  batch_size=args.batch_size, learning_rate=spec['learning_rate'], weight_decay=spec['weight_decay'],
                                  dice_weight=case['dice_weight'], augment=case['augment'], seed=case['seed'],
                                  cpu_threads=args.cpu_threads, device=args.device)
        if directory.exists():
            if not (directory/'frozen-selection.json').exists() and (directory/'last.pth').exists():
                training_args.resume = True
                train(training_args)
            frozen, config = verify_frozen_run(directory)
            expected = {'seed': case['seed'], 'augment': case['augment'], 'dice_weight': case['dice_weight'],
                        'epochs': args.epochs, 'batch_size': args.batch_size, 'image_size': args.image_size,
                        'base_channels': args.base_channels, 'learning_rate': spec['learning_rate'],
                        'weight_decay': spec['weight_decay']}
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Completed run differs from requested protocol: {name}')
            if frozen['cache_fingerprint'] != cache['fingerprint']:
                raise ValueError(f'Completed run used different source data: {name}')
            reused = True
        else:
            train(training_args)
            frozen, _ = verify_frozen_run(directory)
            reused = False
        results.append({'name': name, 'reused': reused, 'checkpoint_sha256': frozen['checkpoint_sha256'],
                        'valid_dice': frozen['valid_dice'], 'threshold': frozen['threshold'], 'seed': frozen['seed']})
        write_json(output/'suite-summary.json', {'status': 'complete' if len(results) == len(requested) else 'running',
                                                'protocol': protocol, 'runs': results, 'test_accessed': False})
    return {'status': 'complete', 'protocol': protocol, 'runs': results, 'test_accessed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--base-channels', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cases', nargs='+')
    result = run_suite(parser.parse_args(argv))
    print(json.dumps({'status': result['status'], 'completed_runs': len(result['runs'])}))


if __name__ == '__main__':
    main()
