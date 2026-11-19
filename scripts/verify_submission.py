"""Audit evidence for the 18 CS406 clauses without assigning an automatic grade.

The default command describes unresolved work and exits successfully. Use
``--claim-ready`` for a gate that returns 1 when mandatory evidence or human
confirmations remain unresolved. ``--package`` audits a SOURCE zip, not a data
or model release. No network requests, model loading, or training are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


KINDS = {'mandatory', 'optional', 'human_confirmation'}
STATES = {'evidence_available', 'confirmed', 'pending', 'planned', 'not_applicable'}
EVIDENCE_TYPES = {'file', 'json', 'checkpoint', 'confirmation'}


def _contained_path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative.strip() or '\x00' in relative
            or PureWindowsPath(relative).drive or relative.startswith(('/', '\\'))):
        raise ValueError(f'Evidence path must be relative: {relative!r}')
    candidate = (root / relative.replace('\\', '/')).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f'Evidence path escapes its declared root: {relative!r}')
    return candidate


def _validate_config(config: dict, roots: dict[str, Path]) -> list[dict]:
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('Expected rubric schema_version 1.')
    if config.get('automatic_grade') is not False:
        raise ValueError('The rubric must explicitly disable automatic_grade.')
    entries = config.get('requirements')
    if not isinstance(entries, list) or not entries:
        raise ValueError('requirements must be a nonempty list.')
    ids, clauses = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Every requirement must be an object.')
        identifier = entry.get('id')
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError('Every requirement needs an id.')
        if identifier in ids:
            raise ValueError(f'Duplicate requirement id: {identifier}')
        ids.add(identifier)
        clause = entry.get('clause')
        if type(clause) is not int or not 1 <= clause <= 18:
            raise ValueError(f'{identifier}: clause must be an integer from 1 to 18.')
        clauses.add(clause)
        if entry.get('kind') not in KINDS or entry.get('state') not in STATES:
            raise ValueError(f'{identifier}: unsupported kind or state.')
        for field in ('statement', 'acceptance'):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f'{identifier}: missing {field}.')
        if entry['state'] == 'not_applicable' and (
                entry['kind'] != 'optional' or not entry.get('reason')):
            raise ValueError(f'{identifier}: only optional entries with a reason may be not_applicable.')
        if not isinstance(entry.get('evidence'), list):
            raise ValueError(f'{identifier}: evidence must be a list.')
        for item in entry['evidence']:
            if not isinstance(item, dict) or item.get('root') not in roots:
                raise ValueError(f'{identifier}: evidence root must be repo or project.')
            _contained_path(roots[item['root']], item.get('path'))
            if item.get('type', 'file') not in EVIDENCE_TYPES:
                raise ValueError(f'{identifier}: unsupported evidence type.')
            if item.get('asset_root', 'project') not in roots:
                raise ValueError(f'{identifier}: invalid checkpoint asset_root.')
            checks = item.get('checks', [])
            if not isinstance(checks, list):
                raise ValueError(f'{identifier}: checks must be a list.')
            for check in checks:
                if (not isinstance(check, dict) or set(check) != {'pointer', 'equals'}
                        or not isinstance(check['pointer'], str)
                        or (check['pointer'] and not check['pointer'].startswith('/'))):
                    raise ValueError(f'{identifier}: use JSON checks {{pointer, equals}}.')
    if clauses != set(range(1, 19)):
        raise ValueError('Rubric coverage must include all 18 DOCX clauses.')
    return entries


def _json_pointer(document, pointer: str):
    value = document
    if pointer:
        for token in pointer[1:].split('/'):
            token = token.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                if not re.fullmatch(r'0|[1-9][0-9]*', token):
                    raise KeyError(pointer)
                value = value[int(token)]
            elif isinstance(value, dict):
                value = value[token]
            else:
                raise KeyError(pointer)
    return value


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        cache[path] = digest.hexdigest()
    return cache[path]


def _check_evidence(item: dict, roots: dict[str, Path], hashes: dict[Path, str]) -> dict:
    path = _contained_path(roots[item['root']], item['path'])
    result = {'root': item['root'], 'path': item['path'], 'status': 'valid', 'issues': []}
    if not path.is_file():
        return {**result, 'status': 'missing', 'issues': ['Evidence file is missing.']}
    issues = result['issues']
    try:
        if path.stat().st_size == 0:
            issues.append('Evidence file is empty.')
        evidence_type = item.get('type', 'file')
        if (evidence_type in {'json', 'checkpoint', 'confirmation'}
                or item.get('checks') or path.suffix.lower() == '.json'):
            document = json.loads(path.read_text(encoding='utf-8-sig'))
            for check in item.get('checks', []):
                try:
                    actual = _json_pointer(document, check['pointer'])
                    # JSON booleans must not silently compare equal to 0 or 1.
                    if json.dumps(actual, sort_keys=True) != json.dumps(check['equals'], sort_keys=True):
                        issues.append(f"JSON {check['pointer']} does not match its declared expectation.")
                except (KeyError, IndexError):
                    issues.append(f"JSON pointer is missing: {check['pointer']}")
            if evidence_type == 'confirmation':
                if (not isinstance(document, dict) or document.get('confirmed') is not True
                        or any(not isinstance(document.get(key), str) or not document[key].strip()
                               for key in ('confirmed_by', 'confirmed_at', 'scope'))):
                    issues.append('Human confirmation needs confirmed=true, confirmed_by, confirmed_at and scope.')
            if evidence_type == 'checkpoint':
                if not isinstance(document, dict):
                    raise ValueError('Checkpoint manifest must be an object.')
                size, digest = document.get('bytes'), document.get('sha256')
                if type(size) is not int or size <= 0 or not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest):
                    raise ValueError('Checkpoint manifest needs positive bytes and a 64-digit SHA-256.')
                asset = _contained_path(roots[item.get('asset_root', 'project')], document.get('relative_path'))
                result['asset'] = str(asset)
                if not asset.is_file():
                    issues.append('Checkpoint asset is missing.')
                elif asset.stat().st_size != size:
                    issues.append('Checkpoint byte size does not match manifest.')
                elif _sha256(asset, hashes) != digest.lower():
                    issues.append('Checkpoint SHA-256 does not match manifest.')
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        issues.append(f'Cannot validate evidence: {error}')
    if issues:
        result['status'] = 'invalid'
    return result


def _check_package(package: Path | None) -> dict:
    result = {'checked': package is not None, 'issues': []}
    if package is None:
        return result
    result['path'] = str(package)
    prohibited_dirs = {'.git', '.venv', 'venv', '__pycache__', '.aws', '.codex',
                       'data', 'imagestrain', 'imagesvalid', 'imagestest',
                       'labelstrain', 'labelsvalid', 'labelstest'}
    try:
        with zipfile.ZipFile(package) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if not members:
                result['issues'].append('Source package contains no files.')
            for member in members:
                name = member.filename.replace('\\', '/')
                path = PurePosixPath(name)
                parts = {part.lower() for part in path.parts}
                reasons = []
                if '..' in parts or path.is_absolute() or PureWindowsPath(name).drive:
                    reasons.append('unsafe member path')
                if parts & prohibited_dirs or path.name.lower() == '.env':
                    reasons.append('dataset, environment or private content')
                if path.suffix.lower() in {'.pth', '.pt', '.ckpt', '.pkl', '.pickle', '.joblib'}:
                    reasons.append('model binary belongs in a model release')
                if member.file_size > 10 * 1024 * 1024:
                    reasons.append('source member exceeds 10 MiB')
                if reasons:
                    result['issues'].append(f'{name}: {"; ".join(reasons)}')
    except (OSError, zipfile.BadZipFile) as error:
        result['issues'].append(f'Cannot inspect source package: {error}')
    return result


def audit_submission(config: dict, *, repo_dir: Path, project_dir: Path,
                     package: Path | None = None) -> dict:
    """Check declared evidence; available prose is not semantic or human certification."""
    roots = {'repo': Path(repo_dir).resolve(), 'project': Path(project_dir).resolve()}
    entries = _validate_config(config, roots)
    counts = {kind: dict.fromkeys(('passed', 'pending', 'failed', 'not_applicable'), 0)
              for kind in sorted(KINDS)}
    records, blocking, human_pending, hashes = [], [], [], {}
    for entry in entries:
        evidence = [_check_evidence(item, roots, hashes) for item in entry['evidence']]
        issues = []
        if entry['state'] == 'not_applicable':
            status = 'not_applicable'
        elif entry['state'] in {'planned', 'pending'}:
            status = 'pending'
            issues.append('Requirement is explicitly unresolved in the evidence ledger.')
        elif not evidence or any(item['status'] != 'valid' for item in evidence):
            status = 'failed'
            issues.append('Every declared evidence file must pass its checks.')
        elif entry['kind'] == 'human_confirmation' and (
                entry['state'] != 'confirmed'
                or not any(item.get('type') == 'confirmation' for item in entry['evidence'])):
            status = 'pending'
            issues.append('A recorded human confirmation is required; file presence cannot replace it.')
        else:
            status = 'passed'
        counts[entry['kind']][status] += 1
        unresolved = status not in {'passed', 'not_applicable'}
        if unresolved and entry['kind'] != 'optional':
            blocking.append(entry['id'])
        if unresolved and entry['kind'] == 'human_confirmation':
            human_pending.append(entry['id'])
        records.append({key: entry[key] for key in ('id', 'clause', 'kind', 'state', 'statement', 'acceptance')}
                       | {'status': status, 'evidence': evidence, 'issues': issues})
    package_result = _check_package(Path(package) if package is not None else None)
    return {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'automatic_grade': False,
        'submission_ready': not blocking and not package_result['issues'],
        'coverage': {'clauses': list(range(1, 19)), 'atomic_requirements': len(records)},
        'roots': {key: str(path) for key, path in roots.items()},
        'counts': counts, 'blocking_ids': blocking, 'human_pending_ids': human_pending,
        'package': package_result, 'requirements': records,
        'limitations': [
            'This is an evidence gate, not a score or a guarantee of rubric satisfaction.',
            'File and JSON checks do not certify prose correctness, rendered layout, provenance truth or oral competence.',
            'Human confirmations are attributed records, not independently authenticated signatures.',
            'Network links are not fetched; verify distribution links and a clean installation separately.',
        ],
    }


def _human_summary(result: dict) -> str:
    lines = ['CS406 submission evidence audit (no automatic grade)',
             f"DOCX clauses covered: 18; atomic entries: {result['coverage']['atomic_requirements']}"]
    for kind, values in result['counts'].items():
        lines.append(f"{kind}: {values['passed']} passed, {values['pending']} pending, {values['failed']} failed")
    lines.append(f"Submission ready: {'yes' if result['submission_ready'] else 'no'}")
    for record in result['requirements']:
        if record['id'] in result['blocking_ids']:
            lines.append(f"  [{record['status']}] {record['id']} (clause {record['clause']}): {record['statement']}")
            for item in record['evidence']:
                if item['status'] != 'valid':
                    lines.append(f"    {item['root']}:{item['path']}: {' '.join(item['issues'])}")
    lines.extend(f'  Package: {issue}' for issue in result['package']['issues'])
    lines.append('Checks establish evidence availability only; human review and source confirmation remain necessary.')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    # Redirected Python output on Windows can default to cp1252. Vietnamese
    # human summaries and ensure_ascii=False JSON must use a stable UTF-8 wire format.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--project-dir', type=Path, default=None)
    parser.add_argument('--requirements', type=Path, default=None)
    parser.add_argument('--package', type=Path, help='Optional SOURCE zip to inspect; no raw data or model binaries.')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON instead of the human summary.')
    parser.add_argument('--output', type=Path, help='Also save the machine-readable audit JSON.')
    parser.add_argument('--claim-ready', action='store_true', help='Return 1 if required evidence or human confirmations are unresolved.')
    args = parser.parse_args(argv)
    repo = args.repo_dir.resolve()
    project = args.project_dir.resolve() if args.project_dir else repo.parent / 'Project'
    requirements = args.requirements or repo / 'config/course-requirements.json'
    try:
        config = json.loads(requirements.read_text(encoding='utf-8-sig'))
        result = audit_submission(config, repo_dir=repo, project_dir=project, package=args.package)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
        print(serialized if args.json else _human_summary(result), end='' if args.json else '\n')
    except (OSError, ValueError, TypeError) as error:
        print(f'Invalid submission audit configuration: {error}', file=sys.stderr)
        return 2
    return 1 if args.claim_ready and not result['submission_ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
