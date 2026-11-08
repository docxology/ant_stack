"""Mermaid source semantics, valid images and preservation on invalid input."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from antstack_core.figures.mermaid import _is_valid_image, preprocess_mermaid_diagrams, sanitize_mermaid_for_rendering, validate_mermaid_syntax


@pytest.mark.parametrize('source',[
    'graph TD; A[Start] -->|rate 0.5| B[End];',
    'graph TD\n subgraph S\n A[friend] --> B[endometriosis]\n end',
    'sequenceDiagram\n a->>b: decision confidence 0.95',
])
def test_valid_diagram_structure_and_edge_labels_are_not_rewritten(source):
    assert sanitize_mermaid_for_rendering(source)==source


def test_invalid_empty_diagram_cannot_remove_existing_images(tmp_path):
    image=tmp_path/'diagram_0123456789abcdef.png'
    Image.new('RGB',(2,2),'white').save(image)
    old=image.read_bytes()
    with pytest.raises(ValueError):
        preprocess_mermaid_diagrams('```mermaid\n\n```',tmp_path,clean_existing=True)
    assert image.read_bytes()==old
    with pytest.raises(ValueError):
        preprocess_mermaid_diagrams('```mermaid\ngraph TD\n A-->B\n```',tmp_path,clean_existing='false')
    assert image.read_bytes()==old


def test_fenced_markdown_examples_are_not_executed(tmp_path):
    source='````markdown\n```mermaid\ngraph TD\n A-->B\n```\n````'
    assert preprocess_mermaid_diagrams(source,tmp_path)==source
    assert not list(tmp_path.iterdir())


def test_png_requires_an_actual_decodable_image(tmp_path):
    file=tmp_path/'image.png'
    file.write_bytes(b'\x89PNG\r\n\x1a\nphony header')
    assert not _is_valid_image(str(file))
    Image.new('RGB',(2,2),'white').save(file)
    assert _is_valid_image(str(file))


def test_structural_preflight_exposes_empty_input_as_invalid():
    result=validate_mermaid_syntax('')
    valid,issues=result
    assert valid is False and issues
