"""Consumer-visible publication/cache/dependency regressions, not mock-success wiring."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from antstack_core.publishing.build_orchestrator import BuildOrchestrator, BuildOrchestratorConfig, BuildTarget
from antstack_core.publishing.pdf_generator import PDFGenerator, PDFGenerationConfig
from antstack_core.publishing.quality_validator import QualityValidator
from antstack_core.publishing.reference_manager import ReferenceManager
from antstack_core.publishing.formatting_fixes import fix_file_paths, fix_naked_urls, fix_variables, process_file
from antstack_core.mathematics import normalize_unicode_math


def _orchestrator(tmp_path):
    return BuildOrchestrator(BuildOrchestratorConfig(build_dir=tmp_path/'build', temp_dir=tmp_path/'temp',
                                                    cache_dir=tmp_path/'cache'))


def test_missing_output_and_declared_inputs_invalidate_cached_build(tmp_path):
    source, config, bibliography, image = [tmp_path/name for name in ('source.md', 'config.yaml', 'refs.bib', 'image.png')]
    for path in (source, config, bibliography, image):
        path.write_text('original')
    output = tmp_path/'paper.pdf'
    output.write_bytes(b'%PDF-1.7\nfixture\n')
    target = BuildTarget('paper', [source], output, build_config={'title':'first'}, dependency_files=[config,bibliography,image])
    target.update_build_hash()
    assert not target.needs_rebuild
    for path in (source, config, bibliography, image):
        path.write_text('changed')
        assert target.needs_rebuild
        target.update_build_hash()
        assert not target.needs_rebuild
    target.build_config['title']='second'
    assert target.needs_rebuild
    target.update_build_hash()
    output.write_bytes(b'%PDF-1.7\nexternally changed\n')
    assert target.needs_rebuild
    target.update_build_hash()
    output.unlink()
    assert target.needs_rebuild


def test_failed_dependency_blocks_dependent_in_parallel_and_sequential_modes(tmp_path):
    for parallel in (False, True):
        orchestrator = _orchestrator(tmp_path/str(parallel))
        orchestrator.add_build_target(BuildTarget('prerequisite', [tmp_path/'missing-a.md'], tmp_path/str(parallel)/'a.pdf'))
        orchestrator.add_build_target(BuildTarget('dependent', [tmp_path/'missing-b.md'], tmp_path/str(parallel)/'b.pdf', dependencies=['prerequisite']))
        results = orchestrator.build_all(parallel=parallel)
        assert not results['prerequisite'].success
        assert not results['dependent'].success
        assert orchestrator.targets['dependent'].build_status == 'blocked'
        assert not orchestrator.targets['dependent'].output_path.exists()


def test_dependency_cycles_and_unknown_targets_fail_before_outputs(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.add_build_target(BuildTarget('a',[tmp_path/'a.md'],tmp_path/'a.pdf',dependencies=['b']))
    with pytest.raises(ValueError):
        orchestrator.build_all()
    orchestrator.add_build_target(BuildTarget('b',[tmp_path/'b.md'],tmp_path/'b.pdf',dependencies=['a']))
    with pytest.raises(ValueError):
        orchestrator.build_target('a')
    assert not (tmp_path/'a.pdf').exists()
    assert not (tmp_path/'b.pdf').exists()


def test_cleanup_preserves_unowned_output_and_caller_temp_contents(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    user_file = orchestrator.config.temp_dir/'user-notes.txt'
    user_file.write_text('unrelated work')
    output = tmp_path/'paper.pdf'
    output.write_bytes(b'user supplied PDF')
    target = BuildTarget('paper',[tmp_path/'source.md'],output)
    orchestrator.add_build_target(target)
    with pytest.raises(ValueError):
        orchestrator.clean_build_artifacts()
    assert output.read_bytes()==b'user supplied PDF'
    assert user_file.read_text()=='unrelated work'
    target.update_build_hash()
    orchestrator.clean_build_artifacts()
    assert not output.exists()
    assert user_file.read_text()=='unrelated work'


def test_missing_sources_and_late_build_errors_cannot_pass_validation(tmp_path):
    missing=QualityValidator().validate_build_readiness([tmp_path/'missing.md'])
    assert not missing.validation_passed and missing.errors > 0
    source=tmp_path/'broken.md'
    source.write_text('# Source\n\\begin{document}\n')
    report=QualityValidator().validate_build_readiness([source])
    assert not report.validation_passed
    assert report.errors==sum(issue.issue_type=='error' for issue in report.issues)
    assert report.total_issues==len(report.issues)
    assert any(issue.category=='build' for issue in report.file_reports[str(source)])


def test_raw_labels_are_actual_definitions_and_code_examples_are_not(tmp_path):
    source=tmp_path/'source.md'
    source.write_text(r'\label{fig:real} Figure~\ref{fig:real}'+'\n```tex\n'+r'\ref{fig:fake}'+'\n```\n')
    report=ReferenceManager().analyze_references([source])
    assert not report.broken_references
    assert len(report.definitions)==1
    assert len(report.usages)==1
    source.write_text(r'\label{fig:real} \ref{fig:reel}')
    manager=ReferenceManager()
    broken=manager.analyze_references([source])
    with pytest.raises(ValueError):
        manager.fix_broken_references(broken,auto_fix=True)
    assert source.read_text()==r'\label{fig:real} \ref{fig:reel}'


def test_native_citation_coverage_rejects_missing_and_duplicate_bibliography(tmp_path):
    source=tmp_path/'source.md'
    source.write_text('See [@present; @missing].\n```markdown\n[@not-a-citation]\n```\n')
    bib=tmp_path/'refs.bib'
    bib.write_text('@article{present, title={Real entry}}\n')
    manager=ReferenceManager()
    report=manager.validate_citations([source],bib)
    assert report['valid'] is False
    assert report['bibliography_validation']['missing_bibliography_entries']==['missing']
    assert manager.validate_citations([source],tmp_path/'absent.bib')['valid'] is False
    bib.write_text('@article{present, title={A}}\n@article{present, title={B}}\n')
    assert manager.validate_citations([source],bib)['valid'] is False


def test_formatting_is_idempotent_and_preserves_structured_sources(tmp_path):
    protected=r'$E_body$ `E_body` ![plot](assets/E_body.png) \href{https://x.test/E_body}{E_body} [@E_body]'
    source='E_body and N_KC, script.py, https://x.test/paper.\n'+protected
    formatted=fix_variables(fix_file_paths(fix_naked_urls(source)))
    assert protected in formatted
    assert r'$E_{\text{body}}$' in formatted
    assert '\t' not in formatted
    assert fix_variables(fix_file_paths(fix_naked_urls(formatted)))==formatted
    path=tmp_path/'source.md'
    path.write_text('Figure ??\n')
    with pytest.raises(ValueError):
        process_file(path)
    assert path.read_text()=='Figure ??\n'


def test_unicode_normalization_does_not_change_resource_identity_or_nested_code():
    source='![α](figures/(set)/α.png) '+r'\href{https://x.test/α}{α}'+' ``α ` b`` '+r'$αx$'
    converted=normalize_unicode_math(source)
    assert '(figures/(set)/α.png)' in converted
    assert r'\href{https://x.test/α}{α}' in converted
    assert '``α ` b``' in converted
    assert r'$\alpha{}x$' in converted
    assert normalize_unicode_math(converted)==converted


def test_zero_exit_renderer_without_current_pdf_preserves_prior_output(tmp_path):
    backend=tmp_path/'renderer'
    backend.write_text('#!'+sys.executable+'\nimport json,sys\n'
                       'if "--to=json" in sys.argv:\n'
                       ' print(json.dumps({"pandoc-api-version":[1,23,1],"meta":{},"blocks":[]}))\n')
    backend.chmod(0o700)
    source=tmp_path/'paper.md'
    source.write_text('# New source\n')
    output=tmp_path/'paper.pdf'
    old=b'%PDF-1.7\nprevious completed output\n'
    output.write_bytes(old)
    result=PDFGenerator(PDFGenerationConfig(pandoc_path=str(backend))).generate_pdf([source],output)
    assert not result.success
    assert output.read_bytes()==old


def test_code_examples_are_not_unmatched_tex_build_environments(tmp_path):
    source=tmp_path/'example.md'
    source.write_text('# Example\n```tex\n\\begin{document}\n```\n')
    report=QualityValidator().validate_build_readiness([source])
    assert report.validation_passed and report.errors==0


def test_bootstrap_interval_outside_mean_is_not_symmetrized_or_clamped(tmp_path):
    from antstack_core.figures.plots import bar_plot
    figure = bar_plot(['observation'], [0.0], confidence_intervals=[(10.0,20.0)], out_path=str(tmp_path/'interval.png'))
    axis=figure.axes[0]
    assert axis.patches[0].get_height()==0.0
    assert axis.collections[0].get_segments()[0].tolist()==[[0.0,10.0],[0.0,20.0]]


def test_native_bare_references_are_checked_without_reading_resource_uris():
    from antstack_core.figures.references import validate_cross_references
    source='# Source\n## Figure: Actual {#fig:actual}\n![Image](local.png)\nSee @fig:actual.\n[URI](https://example.test/?q=@fig:not-a-reference)\n'
    valid=validate_cross_references(source)
    assert valid['valid'] and valid['references']==['fig:actual']
    invalid=validate_cross_references(source+'\nSee @tab:absent.\n')
    assert not invalid['valid']


@pytest.mark.parametrize('labels,values',[([],[]),(['one'],[float('nan')]),(['one','two'],[1.0])])
def test_invalid_bar_observations_cannot_replace_an_existing_output(tmp_path,labels,values):
    from antstack_core.figures.plots import bar_plot
    output=tmp_path/'plot.png'
    original=b'previous owned artifact'
    output.write_bytes(original)
    with pytest.raises(ValueError):
        bar_plot(labels,values,out_path=str(output))
    assert output.read_bytes()==original
