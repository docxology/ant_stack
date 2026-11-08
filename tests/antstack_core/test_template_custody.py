"""Trusted-template failure honesty and output custody."""
from pathlib import Path
import pytest
from antstack_core.publishing.template_engine import TemplateConfig, TemplateEngine

@pytest.mark.parametrize('source', [
    '{{ missing_coefficient }}',
    '{{ include_file("missing.csv") }}',
    '{{ yaml_load("bad: [") }}',
    '{{ json_load("{bad") }}',
])
def test_missing_or_malformed_template_inputs_preserve_previous_document(tmp_path,source):
    (tmp_path/'report.jinja').write_text(source)
    document=tmp_path/'report.md'
    document.write_text('previous reviewed document')
    engine=TemplateEngine(TemplateConfig(tmp_path))
    with pytest.raises(RuntimeError):
        engine.render_template('report',{},document)
    assert document.read_text()=='previous reviewed document'


def test_template_creation_cannot_escape_or_overwrite_existing_source(tmp_path):
    templates=tmp_path/'templates'
    previous=tmp_path/'escaped.jinja'
    previous.write_text('existing source')
    engine=TemplateEngine(TemplateConfig(templates))
    with pytest.raises(ValueError):
        engine.create_publication_template('generic','../escaped')
    assert previous.read_text()=='existing source'


def test_existing_template_is_not_replaced_and_valid_values_render_atomically(tmp_path):
    from antstack_core.publishing.template_engine import render_scientific_document
    template=tmp_path/'reviewed.jinja'
    template.write_text('Energy={{ energy }} J; {{ include_file("units.txt") }}')
    (tmp_path/'units.txt').write_text('per matched window')
    engine=TemplateEngine(TemplateConfig(tmp_path))
    with pytest.raises(FileExistsError):
        engine.create_publication_template('generic','reviewed')
    document=tmp_path/'result.md'
    rendered=render_scientific_document('reviewed',{'energy':2.5},document,template_dir=tmp_path)
    assert rendered=='Energy=2.5 J; per matched window'
    assert document.read_text()==rendered


@pytest.mark.parametrize(('value','sigma','expected'),[
    (1.23456789,1e-9,'1.2345678900'),
    (1234567,25000,'1235000'),
    (3.141592653589793,0,'3.141592653589793'),
])
def test_uncertainty_formatting_preserves_value_at_uncertainty_resolution(tmp_path,value,sigma,expected):
    from decimal import Decimal
    (tmp_path/'precision.jinja').write_text('{{ value | format_uncertainty(sigma,2) }}')
    result=TemplateEngine(TemplateConfig(tmp_path)).render_template('precision',{'value':value,'sigma':sigma})
    rendered_value=result.split('±')[0].strip()
    assert Decimal(rendered_value)==Decimal(expected)


def test_template_syntax_inspection_reports_required_inputs_without_fake_rendering(tmp_path):
    (tmp_path/'report.jinja').write_text('Energy={{ energy }}; {{ include_file("units.txt") }}')
    result=TemplateEngine(TemplateConfig(tmp_path)).validate_template('report')
    assert result['valid'] is True
    assert result['required_variables']==['energy']
    assert result['validation_scope']=='syntax-only'
