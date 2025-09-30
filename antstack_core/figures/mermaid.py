"""Mermaid diagram preprocessing and rendering utilities.

Comprehensive Mermaid diagram handling for scientific publications:
- Mermaid syntax validation and sanitization
- Multi-backend rendering (mmdc, docker, kroki)
- Asset organization and path management
- Publication-quality diagram generation

Following .cursorrules specifications:
- ASCII sanitization for reliable rendering
- Local PNG generation under assets/mermaid/
- Descriptive captions based on diagram content
- Error handling and fallback strategies

References:
- Mermaid.js documentation: https://mermaid.js.org/
- Scientific diagram best practices: https://doi.org/10.1371/journal.pcbi.1003833
"""

from __future__ import annotations
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def validate_mermaid_syntax(mermaid_code: str) -> Dict[str, Union[bool, str, List[str]]]:
    """Validate Mermaid syntax and identify potential issues.
    
    Args:
        mermaid_code: Raw Mermaid diagram code
        
    Returns:
        Dictionary with validation results and suggestions
    """
    issues = []
    suggestions = []
    
    # Check for common syntax issues
    if not mermaid_code.strip():
        issues.append("Empty Mermaid code")
        return {"valid": False, "issues": issues, "suggestions": suggestions}
    
    lines = mermaid_code.strip().split('\n')
    first_line = lines[0].strip()
    
    # Check for diagram type declaration
    diagram_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 
                    'stateDiagram', 'gantt', 'pie', 'erDiagram']
    
    has_diagram_type = any(first_line.startswith(dt) for dt in diagram_types)
    if not has_diagram_type:
        issues.append("Missing diagram type declaration")
        suggestions.append("Start with diagram type (e.g., 'graph TD', 'flowchart LR')")
    
    # Check for problematic Unicode characters
    unicode_chars = ['α', 'β', 'γ', 'Δ', 'δ', 'η', 'μ', 'ρ', '·', '×', '→', '↔']
    for char in unicode_chars:
        if char in mermaid_code:
            issues.append(f"Unicode character '{char}' may cause rendering issues")
            suggestions.append("Replace Unicode with ASCII equivalents")
    
    # Check for LaTeX-style commands
    latex_patterns = [r'\\texttt\{[^}]*\}', r'\$[^$]*\$', r'\\[a-zA-Z]+']
    for pattern in latex_patterns:
        if re.search(pattern, mermaid_code):
            issues.append("LaTeX commands found - may not render in Mermaid")
            suggestions.append("Remove or replace LaTeX formatting")
    
    # Check for proper node termination
    if 'end' in mermaid_code:
        end_lines = [line.strip() for line in lines if line.strip() == 'end']
        if not end_lines:
            issues.append("'end' keyword not on separate line")
            suggestions.append("Put 'end' terminators on their own lines")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
        "line_count": len(lines)
    }


def sanitize_mermaid_for_rendering(mermaid_code: str) -> str:
    """Sanitize Mermaid code for reliable rendering across backends.
    
    Converts problematic characters and formatting to ASCII-safe alternatives
    following the approach used in tools/render_pdf.sh.
    
    Args:
        mermaid_code: Raw Mermaid diagram code
        
    Returns:
        Sanitized Mermaid code safe for rendering
    """
    code = mermaid_code.strip()
    
    # Replace Unicode/math symbols with ASCII equivalents
    replacements = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'Δ': 'Delta', 'δ': 'delta',
        'η': 'eta', 'μ': 'mu', 'ρ': 'rho', '·': '*', '×': 'x',
        '→': '->', '↔': '<->', '𝒪': 'O', '𝑂': 'O', '𝑂(': 'O(', '𝒪(': 'O(',
    }
    
    # Fix invalid Mermaid syntax patterns
    code = re.sub(r'\\,\\mathrm\{A(\d+)\}', r'A\1', code)
    code = re.sub(r'\\,\\mathrm\{([^}]+)\}', r'\1', code)
    
    for unicode_char, ascii_equiv in replacements.items():
        code = code.replace(unicode_char, ascii_equiv)
    
    # Remove LaTeX texttt commands
    code = re.sub(r'\\texttt\{([^}]*)\}', r'\1', code)
    
    # Strip LaTeX inline math markers
    code = code.replace('\\(', '').replace('\\)', '')
    code = code.replace('$', '')
    
    # Remove edge labels with math-like tokens (often problematic)
    code = re.sub(r"\|[^|]*\|", "", code)
    
    # Ensure 'end' terminators are on their own line
    code = re.sub(r"\s+end\s+", "\nend\n", code)
    
    # Collapse multiple spaces to single
    code = re.sub(r"[\t ]+", " ", code)
    
    # Ensure each arrow and node definition separated by newlines
    code = code.replace(";", "\n")
    
    return code


def preprocess_mermaid_diagrams(
    markdown_content: str,
    output_dir: Path,
    img_format: str = "png",
    kroki_url: str = "https://kroki.io",
    clean_existing: bool = True
) -> str:
    """Preprocess Mermaid diagrams in markdown content.
    
    Extracts Mermaid code blocks, renders them to images, and replaces
    the code blocks with image references.
    
    Args:
        markdown_content: Markdown content containing Mermaid blocks
        output_dir: Directory to save rendered images
        img_format: Output image format ('png' or 'svg')
        kroki_url: URL for Kroki rendering service
        clean_existing: If True, clean existing diagrams before processing
        
    Returns:
        Processed markdown with Mermaid blocks replaced by images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean existing diagrams if requested
    if clean_existing:
        for old_file in output_dir.glob("diagram_*"):
            old_file.unlink()
    
    # Find all Mermaid code blocks
    pattern = re.compile(r"```mermaid[^\n]*\n(.*?)\n```", re.DOTALL)
    
    def replace_mermaid_block(match):
        code = match.group(1) or ""
        code = code.strip()
        
        if not code:
            return match.group(0)  # Return original if empty
        
        # Sanitize the code
        clean_code = sanitize_mermaid_for_rendering(code)
        
        # Generate consistent filename based on content hash
        import hashlib
        content_hash = hashlib.md5(clean_code.encode('utf-8')).hexdigest()[:8]
        base_name = f"diagram_{content_hash}"
        
        mmd_path = output_dir / f"{base_name}.mmd"
        img_ext = ".png" if img_format.lower() != "svg" else ".svg"
        img_path = output_dir / f"{base_name}{img_ext}"
        
        # Save the Mermaid code
        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(clean_code)
        
        # Try to render the diagram
        success = _render_mermaid_diagram(str(mmd_path), str(img_path), kroki_url)
        
        if success:
            # Generate relative path for markdown
            rel_path = os.path.relpath(str(img_path), start=os.getcwd())
            
            # Generate descriptive caption based on content
            caption = _generate_diagram_caption(clean_code, base_name)
            
            return f"![{caption}]({rel_path}){{ width=70% }}"
        else:
            # If rendering failed, return original code block
            return match.group(0)
    
    # Replace all Mermaid blocks
    processed_content = pattern.sub(replace_mermaid_block, markdown_content)
    
    return processed_content


def _render_mermaid_diagram(mmd_path: str, output_path: str, kroki_url: str) -> bool:
    """Render a single Mermaid diagram using available backends.
    
    Tries multiple rendering strategies in order of preference:
    1. mmdc (Mermaid CLI) if available
    2. Docker with Mermaid CLI
    3. Kroki web service
    
    Args:
        mmd_path: Path to Mermaid source file
        output_path: Path for output image
        kroki_url: URL for Kroki service
        
    Returns:
        True if rendering succeeded, False otherwise
    """
    # Try mmdc first (fastest if installed)
    if _render_with_mmdc(mmd_path, output_path):
        return True
    
    # Try Docker if available
    if _render_with_docker(mmd_path, output_path):
        return True
    
    # Try Kroki as fallback
    if _render_with_kroki(mmd_path, output_path, kroki_url):
        return True
    
    return False


def _render_with_mmdc(mmd_path: str, output_path: str) -> bool:
    """Render using local mmdc installation."""
    if not shutil.which('mmdc'):
        return False
    
    try:
        cmd = ['mmdc', '-i', mmd_path, '-o', output_path, '-b', 'white']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _is_valid_image(output_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _render_with_docker(mmd_path: str, output_path: str) -> bool:
    """Render using Docker with Mermaid CLI."""
    if not shutil.which('docker'):
        return False
    
    try:
        # Get directory paths for Docker volume mounting
        mmd_dir = os.path.dirname(mmd_path)
        mmd_filename = os.path.basename(mmd_path)
        output_filename = os.path.basename(output_path)
        
        # Get user/group IDs for proper permissions
        uid = os.getuid()
        gid = os.getgid()
        
        cmd = [
            'docker', 'run', '--rm', '-u', f'{uid}:{gid}',
            '-v', f'{mmd_dir}:/data',
            'ghcr.io/mermaid-js/mermaid-cli:latest',
            '-i', f'/data/{mmd_filename}',
            '-o', f'/data/{output_filename}',
            '-b', 'transparent',
            '-w', '4096'
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _is_valid_image(output_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _render_with_kroki(mmd_path: str, output_path: str, kroki_url: str) -> bool:
    """Render using Kroki web service."""
    if not shutil.which('curl'):
        return False
    
    try:
        img_format = 'svg' if output_path.endswith('.svg') else 'png'
        endpoint = f"{kroki_url.rstrip('/')}/mermaid/{img_format}"
        
        with open(output_path, 'wb') as out_f:
            cmd = [
                'curl', '-s', '-X', 'POST',
                '-H', 'Content-Type:text/plain',
                '--data-binary', f'@{mmd_path}',
                endpoint
            ]
            subprocess.run(cmd, check=True, stdout=out_f)
        
        return _is_valid_image(output_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _is_valid_image(path: str) -> bool:
    """Check if file is a valid image."""
    try:
        with open(path, 'rb') as f:
            head = f.read(16)
        
        # Check PNG signature
        if head.startswith(b'\x89PNG\r\n\x1a\n'):
            return True
        
        # Check SVG signature
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                start = f.read(100).lstrip()
            if start.startswith('<svg') or start.startswith('<?xml'):
                return True
        except Exception:
            return False
        
        return False
    except Exception:
        return False


def _generate_diagram_caption(diagram_content: str, diagram_id: str) -> str:
    """Generate descriptive caption based on diagram content."""
    # Extract hash from diagram_id (format: diagram_<hash>)
    diagram_hash = diagram_id.replace('diagram_', '').replace('.mmd', '')
    
    # Analyze content to determine caption type
    if 'Analysis[' in diagram_content and 'Methods[' in diagram_content:
        return ('Enhanced analysis pipeline overview showing analysis modules, '
               'computational methods, orchestration scripts, validation framework, '
               'and empirical scaling results.')
    elif 'Body[' in diagram_content and 'Brain[' in diagram_content:
        return ('Module complexity overview detailing AntBody contact dynamics, '
               'AntBrain sparse neural networks, and AntMind bounded rational '
               'processing pipeline.')
    elif 'Physical[' in diagram_content and 'Control[' in diagram_content:
        return ('Energy flows overview across physical layer (terrain, sensors, '
               'mechanics), control layer (real-time processing), energy analysis '
               'components, and theoretical limits framework.')
    else:
        return f'Computational architecture diagram ({diagram_hash[:4]})'
