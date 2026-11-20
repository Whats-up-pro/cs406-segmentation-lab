"""Submission claims must fail when their underlying evidence is missing or false."""

import copy
import hashlib
import io
import json
import sys
import zipfile

import pytest


def rubric(tmp_path):
    repo = tmp_path / 'repo'
    project = tmp_path / 'Project'
    repo.mkdir()
    project.mkdir()
    (repo / 'README.md').write_text('Reproducible example', encoding='utf-8')
    entries = [
        {'id': f's{clause:02d}', 'clause': clause, 'kind': 'mandatory',
         'state': 'evidence_available', 'statement': 'Documented requirement',
         'acceptance': 'The evidence is available.',
         'evidence': [{'root': 'repo', 'path': 'README.md'}]}
        for clause in range(1, 19)
    ]
    return {'schema_version': 1, 'automatic_grade': False, 'requirements': entries}, repo, project


def audit(config, repo, project, **kwargs):
    from scripts.verify_submission import audit_submission
    return audit_submission(config, repo_dir=repo, project_dir=project, **kwargs)


def test_missing_mandatory_evidence_prevents_submission_ready(tmp_path):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0]['evidence'] = [{'root': 'project', 'path': 'missing.pdf'}]
    result = audit(config, repo, project)
    assert result['submission_ready'] is False
    assert result['blocking_ids'] == ['s01']
    assert result['requirements'][0]['evidence'][0]['status'] == 'missing'
    assert result['automatic_grade'] is False


def test_complete_evidence_can_pass_without_assigning_a_grade(tmp_path):
    config, repo, project = rubric(tmp_path)
    result = audit(config, repo, project)
    assert result['submission_ready'] is True
    assert result['counts']['mandatory']['passed'] == 18
    assert 'grade' not in result


def test_json_status_is_checked_instead_of_trusting_file_presence(tmp_path):
    config, repo, project = rubric(tmp_path)
    (project / 'run.json').write_text('{"status":"running","test":{"images":1}}', encoding='utf-8')
    config['requirements'][0]['evidence'] = [
        {'root': 'project', 'path': 'run.json', 'type': 'json',
         'checks': [{'pointer': '/status', 'equals': 'complete'},
                    {'pointer': '/test/images', 'equals': 295}]}
    ]
    result = audit(config, repo, project)
    assert result['submission_ready'] is False
    assert result['requirements'][0]['evidence'][0]['status'] == 'invalid'


def test_existing_files_do_not_promote_a_planned_run_to_complete(tmp_path):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0]['state'] = 'planned'
    result = audit(config, repo, project)
    assert result['blocking_ids'] == ['s01']
    assert result['requirements'][0]['status'] == 'pending'


def test_optional_and_human_confirmation_have_different_readiness_effects(tmp_path):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0].update(kind='optional', state='planned', evidence=[])
    config['requirements'][1].update(kind='human_confirmation', state='pending', evidence=[])
    result = audit(config, repo, project)
    assert result['blocking_ids'] == ['s02']
    assert result['human_pending_ids'] == ['s02']
    assert result['counts']['optional']['pending'] == 1


def test_human_confirmation_requires_an_attributed_record_not_a_readme(tmp_path):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0].update(kind='human_confirmation', state='confirmed')
    result = audit(config, repo, project)
    assert result['blocking_ids'] == ['s01']
    (project / 'confirmation.json').write_text(json.dumps({
        'confirmed': True, 'confirmed_by': 'Student', 'confirmed_at': '2026-11-19',
        'scope': 'Verified actual team membership',
    }), encoding='utf-8')
    config['requirements'][0]['evidence'] = [
        {'root': 'project', 'path': 'confirmation.json', 'type': 'confirmation'}
    ]
    assert audit(config, repo, project)['submission_ready'] is True


def test_checkpoint_manifest_requires_matching_asset_size_and_hash(tmp_path):
    config, repo, project = rubric(tmp_path)
    data = b'actual synthetic model bytes'
    (project / 'model.pth').write_bytes(data)
    manifest = {'relative_path': 'model.pth', 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}
    (repo / 'checkpoint.json').write_text(json.dumps(manifest), encoding='utf-8')
    config['requirements'][0]['evidence'] = [
        {'root': 'repo', 'path': 'checkpoint.json', 'type': 'checkpoint', 'asset_root': 'project'}
    ]
    assert audit(config, repo, project)['submission_ready'] is True
    (project / 'model.pth').write_bytes(b'x' * len(data))
    result = audit(config, repo, project)
    assert result['blocking_ids'] == ['s01']
    assert 'SHA-256' in ' '.join(result['requirements'][0]['evidence'][0]['issues'])


@pytest.mark.parametrize('path', ['../outside.txt', '/outside.txt', 'C:\\outside.txt'])
def test_evidence_cannot_escape_its_declared_root(tmp_path, path):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0]['evidence'] = [{'root': 'repo', 'path': path}]
    with pytest.raises(ValueError, match='relative|escape'):
        audit(config, repo, project)


def test_source_package_rejects_dataset_weights_and_private_files(tmp_path):
    config, repo, project = rubric(tmp_path)
    package = tmp_path / 'source.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr('source/README.md', 'Run instructions')
        archive.writestr('source/data/imagesTrain/image.png', b'image')
        archive.writestr('source/models/model.pth', b'weights')
        archive.writestr('source/.env', 'TOKEN=example')
    result = audit(config, repo, project, package=package)
    assert result['submission_ready'] is False
    assert len(result['package']['issues']) == 3
    assert result['package']['checked'] is True


def test_empty_and_malformed_json_evidence_cannot_pass(tmp_path):
    config, repo, project = rubric(tmp_path)
    (project / 'empty.txt').touch()
    (project / 'invalid.json').write_text('{broken', encoding='utf-8')
    config['requirements'][0]['evidence'] = [{'root': 'project', 'path': 'empty.txt'}]
    config['requirements'][1]['evidence'] = [{'root': 'project', 'path': 'invalid.json', 'type': 'json'}]
    assert audit(config, repo, project)['blocking_ids'] == ['s01', 's02']


def test_invalid_rubric_cannot_hide_clauses_or_repeat_ids(tmp_path):
    config, repo, project = rubric(tmp_path)
    incomplete = copy.deepcopy(config)
    incomplete['requirements'].pop()
    with pytest.raises(ValueError, match='18|coverage'):
        audit(incomplete, repo, project)
    config['requirements'][1]['id'] = 's01'
    with pytest.raises(ValueError, match='Duplicate'):
        audit(config, repo, project)


def test_cli_emits_machine_audit_and_rejects_unresolved_ready_claim(tmp_path, capsys):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0].update(kind='human_confirmation', state='pending', evidence=[])
    config_file = repo / 'rubric.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')
    audit(config, repo, project)
    from scripts.verify_submission import main
    arguments = ['--repo-dir', str(repo), '--project-dir', str(project),
                 '--requirements', str(config_file), '--json']
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)['submission_ready'] is False
    assert main(arguments + ['--claim-ready']) == 1
    assert json.loads(capsys.readouterr().out)['human_pending_ids'] == ['s01']


def test_windows_legacy_stdout_does_not_turn_vietnamese_audit_into_configuration_error(tmp_path, monkeypatch):
    config, repo, project = rubric(tmp_path)
    config['requirements'][0].update(state='pending', statement='Yêu cầu phân đoạn ảnh')
    config_file = repo / 'rubric.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')
    audit(config, repo, project)
    from scripts.verify_submission import main
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding='cp1252')
    monkeypatch.setattr(sys, 'stdout', stream)
    exit_code = main(['--repo-dir', str(repo), '--project-dir', str(project),
                      '--requirements', str(config_file), '--claim-ready'])
    stream.flush()
    assert exit_code == 1
    assert 'Yêu cầu phân đoạn ảnh' in buffer.getvalue().decode('utf-8')
