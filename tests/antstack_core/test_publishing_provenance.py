"""Real Git/source identity and atomic receipt preservation regressions."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess

import pytest

from antstack_core.publishing.provenance import _git_snapshot, atomic_json, detect_git_state


def _git(root, *args):
    environment=dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    return subprocess.run(['git','-c','commit.gpgsign=false','-c','core.hooksPath='+str(root/'empty-hooks'),*args],
                          cwd=root,env=environment,stdin=subprocess.DEVNULL,capture_output=True,text=True,check=True,timeout=20)


def test_staged_and_untracked_source_bytes_are_dirty_and_bound(tmp_path):
    _git(tmp_path,'init','--template=','-q')
    source=tmp_path/'source.py'
    source.write_text('initial implementation\n')
    _git(tmp_path,'add','source.py')
    _git(tmp_path,'-c','user.name=Ant Stack fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
    source.write_text('staged implementation\n')
    _git(tmp_path,'add','source.py')
    untracked=tmp_path/'new.py'
    untracked.write_text('untracked implementation\n')
    snapshot=_git_snapshot(tmp_path)
    assert snapshot['dirty'] is True
    changes={entry['path']:entry for entry in snapshot['changes']}
    assert changes['source.py']['status']=='M '
    assert changes['new.py']['status']=='??'
    assert changes['source.py']['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    old=changes['new.py']['sha256']
    untracked.write_text('different untracked implementation\n')
    current={entry['path']:entry for entry in _git_snapshot(tmp_path)['changes']}
    assert current['new.py']['sha256']!=old


def test_unborn_git_head_is_unavailable_not_clean(tmp_path):
    _git(tmp_path,'init','--template=','-q')
    assert detect_git_state(tmp_path)==(None,None)


def test_atomic_metadata_preserves_private_mode_and_prior_bytes_on_failure(tmp_path):
    target=tmp_path/'receipt.json'
    target.write_text('{"old":true}\n')
    target.chmod(0o600)
    atomic_json(target,{'new':True})
    assert stat.S_IMODE(target.stat().st_mode)==0o600
    previous=target.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(target,{'invalid':float('nan')})
    assert target.read_bytes()==previous
    link=tmp_path/'alias.json'
    link.symlink_to(target)
    with pytest.raises(ValueError):
        atomic_json(link,{'overwrite':True})
    assert target.read_bytes()==previous


def test_provenance_does_not_scrape_parent_process_credentials(tmp_path,monkeypatch):
    import json
    import sys
    from antstack_core.publishing.provenance import build_run_provenance
    monkeypatch.setattr(sys,'argv',['parent-script','--token','PRIVATE_TEST_SENTINEL'])
    record=build_run_provenance(cwd=tmp_path)
    assert 'PRIVATE_TEST_SENTINEL' not in json.dumps(record.to_dict())


def test_atomic_text_is_private_before_publication(tmp_path, monkeypatch):
    import antstack_core.publishing.provenance as module
    target = tmp_path / "private.txt"
    target.write_text("prior")
    target.chmod(0o600)
    publish = module.publish_file
    def inspect(source, destination):
        assert stat.S_IMODE(source.stat().st_mode) == 0o600
        return publish(source, destination)
    monkeypatch.setattr(module, "publish_file", inspect)
    previous = os.umask(0o022)
    try:
        module.atomic_text(target, "private replacement")
    finally:
        os.umask(previous)
    assert target.read_text() == "private replacement"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_cross_filesystem_publication_preserves_bytes_and_mode(tmp_path, monkeypatch):
    import errno
    import antstack_core.publishing.provenance as module
    source_dir, target_dir = tmp_path / "staging", tmp_path / "mounted-output"
    source_dir.mkdir(); target_dir.mkdir()
    source, target = source_dir / "fresh.pdf", target_dir / "final.pdf"
    source.write_bytes(b"%PDF-new produced bytes")
    target.write_bytes(b"prior"); target.chmod(0o600)
    replace = module.os.replace
    def cross_device(first, second):
        if Path(first).parent != Path(second).parent:
            raise OSError(errno.EXDEV, "Simulated distinct mounted filesystem")
        return replace(first, second)
    monkeypatch.setattr(module.os, "replace", cross_device)
    module.publish_file(source, target)
    assert target.read_bytes() == b"%PDF-new produced bytes"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_tex_resource_arguments_are_not_citations(tmp_path):
    from antstack_core.publishing.reference_manager import ReferenceManager
    source, bibliography = tmp_path / "source.md", tmp_path / "references.bib"
    source.write_text(r"\href{https://example.test/?q=@draft}{See @real} \url{https://example.test/@other} \includegraphics[width=.5\textwidth]{assets/@image.png} \cite{real}")
    bibliography.write_text("@article{real, title={Real}, author={Example, A}, year={2020}}")
    result = ReferenceManager().validate_citations([source], bibliography)
    assert result["valid"] is True
    assert result["citations_found"] == ["real"]


def test_tex_resource_mask_preserves_document_reference_identity():
    from antstack_core.figures.references import CrossReferenceValidator
    content = "# One section {#sec:one}\n\n" + r"\href{https://example.test/?q=@draft}{Search}" + "\n" + r"\url{https://example.test/@other}" + "\n" + r"See \ref{sec:one}."
    report = CrossReferenceValidator().validate_document(content)
    assert report["valid"] is True
    assert report["definitions"] == ["sec:one"]
    assert report["references"] == ["sec:one"]
