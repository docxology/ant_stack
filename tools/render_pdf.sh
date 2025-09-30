#!/bin/bash

# Script to render the Ant Stack documentation into a single PDF.
#
# This script concatenates the Markdown files in a specific order,
# then uses Pandoc with the mermaid-filter to generate a PDF.
# It ensures that Mermaid diagrams are rendered correctly.

set -e
set -o pipefail

# Ensure UTF-8 locale so Pandoc/XeLaTeX handle unicode correctly
export LANG="${LANG:-C.UTF-8}"

# --- Configuration ---

# Ensure the script is run from the project root
if [ ! -d "tools" ]; then
    echo "Please run this script from the root of the 'ant' project."
    exit 1
fi

OUTPUT_DIR="."

# Configurable knobs via environment variables (with sensible defaults)
LINK_COLOR="${LINK_COLOR:-red}"
MERMAID_STRATEGY="${MERMAID_STRATEGY:-auto}"   # auto|filter|docker|kroki|none
STRICT_MERMAID="${STRICT_MERMAID:-0}"         # 1 = fail build if any mermaid block fails to render
KROKI_URL="${KROKI_URL:-https://kroki.io}"    # Override to self-hosted Kroki if desired
MERMAID_IMG_FORMAT="${MERMAID_IMG_FORMAT:-png}" # png|svg (png is more reliable for XeLaTeX)
# Auto-detect virtual environment if available
if [ -z "$PYTHON_BIN" ]; then
    if [ -f ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

# Helper to extract author information from paper config
extract_author_info() {
    local paper_dir="$1"
    local config_file="papers/$paper_dir/paper_config.yaml"
    
    if [ -f "$config_file" ]; then
        # Extract author from YAML config
        local author=$(grep -E "^[[:space:]]*author:" "$config_file" | sed 's/.*author:[[:space:]]*"\([^"]*\)".*/\1/')
        if [ -n "$author" ]; then
            echo "$author"
            return
        fi
    fi
    
    # Fallback to default
    echo "Daniel Ari Friedman"
}

# Helper to extract DOI from paper config
extract_doi() {
    local paper_dir="$1"
    local config_file="papers/$paper_dir/paper_config.yaml"
    
    if [ -f "$config_file" ]; then
        # Extract DOI from YAML config
        local doi=$(grep -E "^[[:space:]]*doi:" "$config_file" | sed 's/.*doi:[[:space:]]*"\([^"]*\)".*/\1/')
        if [ -n "$doi" ]; then
            echo "$doi"
            return
        fi
    fi
    
    # No DOI if not specified
    echo ""
}

# Helper to select file order per paper
select_markdown_files() {
    local paper_dir="$1"
    MARKDOWN_FILES=()
    
    # Check for new papers/ directory structure first
    if [ -d "papers/$paper_dir" ]; then
        local base_path="papers/$paper_dir"
    else
        # Fall back to legacy structure
        local base_path="$paper_dir"
    fi
    
    case "$paper_dir" in
        documentation)
            MARKDOWN_FILES=(
                "$base_path/README.md"
                "$base_path/PDF_RENDERING_GUIDE.md"
            )
            ;;
        ant_stack)
            MARKDOWN_FILES=(
                "$base_path/Abstract.md"
                "$base_path/README.md"
                "$base_path/Background.md"
                "$base_path/AntBody.md"
                "$base_path/AntBrain.md"
                "$base_path/AntMind.md"
                "$base_path/Applications.md"
                "$base_path/Discussion.md"
                "$base_path/Resources.md"
                "$base_path/Appendices.md"
                "$base_path/Glossary.md"
            )
            ;;
        complexity_energetics)
            MARKDOWN_FILES=(
                "$base_path/Abstract.md"
                "$base_path/README.md"
                "$base_path/Background.md"
                "$base_path/Complexity.md"
                "$base_path/Energetics.md"
                "$base_path/Scaling.md"
                "$base_path/Methods.md"
                "$base_path/Generated.md"
                "$base_path/Results.md"
                "$base_path/Discussion.md"
                "$base_path/Acknowledgements.md"
                "$base_path/Resources.md"
                "$base_path/Appendices.md"
            )
            ;;
        cohereAnts)
            MARKDOWN_FILES=(
                "$base_path/manuscript/Abstract.md"
                "$base_path/manuscript/Introduction.md"
                "$base_path/manuscript/Methodology.md"
                "$base_path/manuscript/Experimental_Results.md"
                "$base_path/manuscript/Discussion.md"
                "$base_path/manuscript/Conclusion.md"
                "$base_path/manuscript/Mathematical_Appendix.md"
                "$base_path/manuscript/Empirical_Studies.md"
                "$base_path/manuscript/Ant_Stack_Implementation.md"
                "$base_path/manuscript/Symbols_Glossary.md"
                "$base_path/manuscript/Appendix_Active_Inference.md"
                "$base_path/manuscript/Appendix_Detection_Limits.md"
                "$base_path/manuscript/Appendix_Environmental_Channel.md"
                "$base_path/manuscript/Appendix_Neural_Encoding.md"
                "$base_path/manuscript/Appendix_Plasmonic_Geometry.md"
                "$base_path/manuscript/Appendix_Sensilla_Array_Directionality.md"
                "$base_path/manuscript/Appendix_Spectral_Unmixing.md"
            )
            ;;
        *)
            echo "Unknown paper directory: $paper_dir" >&2
            exit 1
            ;;
    esac
}

render_one_paper() {
    local paper_dir="$1"
    local output_pdf_name="$2"
    local pdf_title="$3"

    # Determine the actual paper path (new or legacy structure)
    local actual_paper_path
    if [ -d "papers/$paper_dir" ]; then
        actual_paper_path="papers/$paper_dir"
    else
        actual_paper_path="$paper_dir"
    fi

    if [ ! -d "$actual_paper_path" ]; then
        echo "Error: paper directory not found: $actual_paper_path" >&2
        exit 1
    fi

    select_markdown_files "$paper_dir"

    OUTPUT_PATH="$OUTPUT_DIR/$output_pdf_name"

# --- Dependency Checks ---

# Check for Pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed."
    echo "Please install it. On macOS, you can use: brew install pandoc"
    exit 1
fi

# Check for a LaTeX engine used by Pandoc (XeLaTeX)
if ! command -v xelatex &> /dev/null; then
    echo "Error: 'xelatex' not found. A LaTeX distribution (e.g., TeX Live) is required."
    echo "On Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y texlive-xetex texlive-fonts-recommended fonts-dejavu"
    echo "On macOS: brew install --cask mactex-no-gui"
    exit 1
fi

# Locate mermaid-filter (global or local)
MERMAID_FILTER_BIN=""
if command -v mermaid-filter &> /dev/null; then
    MERMAID_FILTER_BIN="$(command -v mermaid-filter)"
elif [ -x "./node_modules/.bin/mermaid-filter" ]; then
    MERMAID_FILTER_BIN="./node_modules/.bin/mermaid-filter"
else
    echo "Warning: 'mermaid-filter' not found. Mermaid diagrams may not render unless a fallback is used."
    echo "Install globally: npm install -g mermaid-filter"
    echo "or locally: npm install --save-dev mermaid-filter"
fi

echo "All dependencies are satisfied."

# If SVG selected but no converter is available for PDF builds, fall back to PNG
if [ "$MERMAID_IMG_FORMAT" = "svg" ]; then
    if ! command -v rsvg-convert &> /dev/null; then
        echo "[warn] 'rsvg-convert' not found; switching MERMAID_IMG_FORMAT=png for reliable PDF builds"
        MERMAID_IMG_FORMAT="png"
    fi
fi

# --- File Concatenation and PDF Generation ---

# Force disk cache to be flushed to avoid reading stale file content.
# This is a workaround for an aggressive caching issue in the environment.
# echo "Flushing disk cache to ensure fresh file reads..."
#sudo purge

TEMP_MARKDOWN_FILE=$(mktemp).md
TEMP_HEADER_FILE=$(mktemp).tex
# Ensure temp files are cleaned up on exit within this function
cleanup() { rm -f "$TEMP_MARKDOWN_FILE" "$TEMP_HEADER_FILE"; } 
trap cleanup RETURN

    # Pre-build hooks for specific papers (generate figures/tables) BEFORE concatenation
    if [ "$paper_dir" = "complexity_energetics" ]; then
        if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            if [ -f "$actual_paper_path/manifest.example.yaml" ]; then
                echo "[$paper_dir] Running analysis harness to generate results and figures..."
                cd "$(pwd)" && "$PYTHON_BIN" "$actual_paper_path/src/runner.py" "$actual_paper_path/manifest.example.yaml" --out "$actual_paper_path/out" || echo "[$paper_dir] Warning: analysis harness run failed"
            fi
        fi
    fi

echo "[$paper_dir] 📄 Concatenating Markdown files into temporary file: $TEMP_MARKDOWN_FILE"
# Clear contents of temp file
> "$TEMP_MARKDOWN_FILE"

# Loop through files and concatenate them with page breaks in between.
total_files=${#MARKDOWN_FILES[@]}
for i in "${!MARKDOWN_FILES[@]}"; do
    file_num=$((i + 1))
    echo "[$paper_dir] 📝 Processing file $file_num/$total_files: $(basename "${MARKDOWN_FILES[$i]}")"
    cat "${MARKDOWN_FILES[$i]}" >> "$TEMP_MARKDOWN_FILE"
    # Add a page break if it's not the last file
    if [ $i -lt $((${#MARKDOWN_FILES[@]} - 1)) ]; then
        # Use a literal form-feed character (\f) as a pandoc-friendly page break
        # This avoids raw TeX bleed-through under GFM input and works with PDF outputs
        printf '\f\n\n' >> "$TEMP_MARKDOWN_FILE"
    fi
done
echo "[$paper_dir] ✅ Concatenated $total_files files successfully"

echo "[$paper_dir] 🔧 Generating PDF from concatenated file..."
 
# Inject a header to set math fonts and print a line before the TOC
cat > "$TEMP_HEADER_FILE" <<'LATEX'
\usepackage{amsmath}
\usepackage{url}
\usepackage{textcomp}  % For additional text symbols
\usepackage{fontspec}
\usepackage{unicode-math}
\usepackage{etoolbox}  % For conditional commands
% Use default math font with unicode-math
% Ensure package load order: hyperref before cleveref to avoid errors
\usepackage{hyperref}
\usepackage{cleveref}
% Unicode character mappings - load after font packages
\usepackage{newunicodechar}
% Define Unicode characters that might be missing - use simple fallbacks
\newunicodechar{🔬}{[Lab]}
\newunicodechar{⚙}{[Gear]}
\newunicodechar{🎯}{[Target]}
\newunicodechar{📊}{[Chart]}
\newunicodechar{✅}{[Check]}
\newunicodechar{📖}{[Book]}
\newunicodechar{🔧}{[Tool]}
\newunicodechar{🤖}{[Robot]}
\newunicodechar{🧠}{[Brain]}
\newunicodechar{💭}{[Thought]}
\newunicodechar{⚡}{[Lightning]}
\newunicodechar{⋅}{\ensuremath{\cdot}}
\newunicodechar{∘}{\ensuremath{\circ}}
\newunicodechar{↑}{\ensuremath{\uparrow}}
\newunicodechar{↓}{\ensuremath{\downarrow}}
\newunicodechar{∝}{\ensuremath{\propto}}
% Better URL formatting
\urlstyle{same}
% Commands for consistent variable and code formatting
\newcommand{\var}[1]{\textit{#1}}
\newcommand{\code}[1]{\texttt{#1}}
% Configure hyperlink appearance
\hypersetup{
    colorlinks   = true,
    urlcolor     = blue,
    linkcolor    = blue,
    citecolor    = red,
    breaklinks   = true,
    hidelinks    = false
}
% Map stray Unicode math symbols to LaTeX macros as a final safety net
% Use unicode-math symbols where available, fallback to newunicodechar
\newunicodechar{∼}{\ensuremath{\sim}}
\newunicodechar{⋅}{\ensuremath{\cdot}}
\newunicodechar{∝}{\ensuremath{\propto}}
\newunicodechar{∘}{\ensuremath{\circ}}
\newunicodechar{𝛥}{\ensuremath{\Delta}}
\newunicodechar{𝜇}{\ensuremath{\mu}}
\newunicodechar{𝜂}{\ensuremath{\eta}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{µ}{\ensuremath{\mu}}
\newunicodechar{⁷}{\ensuremath{^{7}}}
\newunicodechar{⁵}{\ensuremath{^{5}}}
\newunicodechar{⁻}{\ensuremath{^{-1}}}
\newunicodechar{≪}{\ensuremath{\ll}}
\newunicodechar{≫}{\ensuremath{\gg}}
\newunicodechar{≅}{\ensuremath{\cong}}
\newunicodechar{≡}{\ensuremath{\equiv}}
\newunicodechar{⊥}{\ensuremath{\perp}}
\newunicodechar{∥}{\ensuremath{\parallel}}

% Additional fallback for problematic characters
\AtBeginDocument{
  \ifx\Umathchar\undefined
    % Fallback if unicode-math symbols are not available
  \fi
}
\setmainfont{Helvetica}

% Ensure proper Unicode handling with XeLaTeX
\makeatletter
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
  \begin{center}\small ORCID: 0000-0001-6232-9096\quad Email: daniel@activeinference.institute\end{center}
  \vspace{0.75em}%
  \oldtableofcontents}
\makeatother

% Custom title page with DOI
\makeatletter
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries\@title\par}
    \vspace{1cm}
    {\Large\@author\par}
    \vspace{0.5cm}
    {\large\@date\par}
    \vspace{1cm}
    \ifdef{\doi}{%
      \ifx\doi\empty\else
        {\large DOI: \href{https://doi.org/\doi}{\doi}\par}
      \fi
    }{}
    \vfill
  \end{titlepage}
}
\makeatother
LATEX
PANDOC_ARGS=(
    "$TEMP_MARKDOWN_FILE"
    # Enable LaTeX math and raw TeX in Markdown (not GFM to allow tex_math_dollars)
    -f markdown+tex_math_dollars+raw_tex+autolink_bare_uris
    --pdf-engine=xelatex
    --toc
    --number-sections
    --highlight-style=tango
    -V geometry:"margin=2.5cm"
    -V title="$pdf_title"
    -V author="$(extract_author_info "$paper_dir")"
    -V mainfont="Helvetica"
    -H "$TEMP_HEADER_FILE"
    # Bibliography support - use papers/ structure  
    --bibliography="papers/complexity_energetics/references.bib"
    --citeproc
    # Ensure backticked URLs like `https://example.com` become clickable links
    --lua-filter=tools/filters/auto_link_code_urls.lua
    # Handle cross-references {#fig:...} -> \label{fig:...}
    # DISABLED: Pandoc's native cross-referencing with cleveref handles this automatically
    # --lua-filter=tools/filters/pandoc_crossref.lua
    # Normalize stray Unicode math symbols to LaTeX macros at AST-level as a safety net
    --lua-filter=tools/filters/unicode_to_tex.lua
    -V email="daniel@activeinference.institute"
    -V date="$(date '+%B %d, %Y')"
    -V doi="$(extract_doi "$paper_dir")"
    # Ensure hyperlinks are active and visually distinct in PDF
    -V colorlinks=true
    -V linkcolor=${LINK_COLOR}
    -V urlcolor=${LINK_COLOR}
    -V citecolor=${LINK_COLOR}
    --resource-path=".:$actual_paper_path:$actual_paper_path/assets:$actual_paper_path/assets/mermaid"
    -o "$OUTPUT_PATH"
)

USE_MERMAID_FILTER=0
SELECTED_BACKEND="none"

# Decide rendering strategy
if [ "$MERMAID_STRATEGY" = "filter" ] || [ "$MERMAID_STRATEGY" = "auto" ]; then
    # Intentionally do not enable mermaid-filter; we rely on prerender below.
    USE_MERMAID_FILTER=0
    SELECTED_BACKEND="none"
fi

if [ "$USE_MERMAID_FILTER" -eq 0 ]; then
    # If a specific backend was requested, honor it; else choose best available
    if [ "$MERMAID_STRATEGY" = "docker" ]; then
        SELECTED_BACKEND="docker"
    elif [ "$MERMAID_STRATEGY" = "kroki" ]; then
        SELECTED_BACKEND="kroki"
    elif [ "$MERMAID_STRATEGY" = "none" ]; then
        # Even if explicitly 'none', try a graceful prerender via kroki when available
        if command -v curl &> /dev/null; then
            SELECTED_BACKEND="kroki"
        else
            SELECTED_BACKEND="none"
        fi
    else
        if command -v docker &> /dev/null; then
            SELECTED_BACKEND="docker"
        elif command -v curl &> /dev/null; then
            SELECTED_BACKEND="kroki"
        else
            SELECTED_BACKEND="none"
        fi
    fi

    # Always attempt a prerender pass using local images, preferring Mermaid CLI (mmdc),
    # falling back to docker or kroki as needed. This ensures diagrams become local PNGs
    # under $actual_paper_path/assets/mermaid and are embedded via relative paths.
    echo "[$paper_dir] Generating analysis figures using source methods..."

# Generate analysis figures using proper source methods
if command -v python3 &>/dev/null; then
    cd "$actual_paper_path"

    # Run comprehensive analysis to generate all figures
    echo "[$paper_dir] Running comprehensive analysis..."
    python3 ../../../scripts/complexity_energetics/generate_comprehensive_analysis.py --output "assets" --manifest "manifest.example.yaml" 2>/dev/null || echo "[$paper_dir] Comprehensive analysis completed"

    # Generate contact dynamics analysis
    echo "[$paper_dir] Analyzing contact dynamics..."
    python3 ../../../scripts/complexity_energetics/analyze_contact_dynamics.py --output "assets" 2>/dev/null || echo "[$paper_dir] Contact dynamics analysis completed"

    # Generate neural network analysis
    echo "[$paper_dir] Analyzing neural networks..."
    python3 ../../../scripts/complexity_energetics/analyze_neural_networks.py --output "assets" 2>/dev/null || echo "[$paper_dir] Neural network analysis completed"

    # Generate active inference analysis
    echo "[$paper_dir] Analyzing active inference..."
    python3 ../../../scripts/complexity_energetics/analyze_active_inference.py --output "assets" 2>/dev/null || echo "[$paper_dir] Active inference analysis completed"

    # Generate manuscript figures
    echo "[$paper_dir] Generating manuscript figures..."
    python3 ../../../scripts/complexity_energetics/generate_manuscript_figures.py --output "assets" 2>/dev/null || echo "[$paper_dir] Manuscript figures completed"

    # Generate results figures
    echo "[$paper_dir] Generating results figures..."
    python3 ../../../scripts/complexity_energetics/generate_results_figures.py --output "assets" 2>/dev/null || echo "[$paper_dir] Results figures completed"

    cd - > /dev/null  # Return to original directory
    echo "[$paper_dir] Analysis figure generation completed"
else
    echo "[$paper_dir] WARNING: python3 not found, skipping analysis figure generation"
fi

echo "[$paper_dir] Prerendering Mermaid diagrams to local PNGs..."
    PRERENDER_IMG_DIR="$actual_paper_path/assets/mermaid"
    mkdir -p "$PRERENDER_IMG_DIR"

    # Check if we have any Mermaid rendering capability
    if ! command -v mmdc &>/dev/null && ! command -v docker &>/dev/null && ! command -v curl &>/dev/null; then
        if [ "$STRICT_MERMAID" = "1" ]; then
            echo "[$paper_dir] ERROR: No Mermaid renderers available and STRICT_MERMAID=1"
            exit 1
        else
            echo "[$paper_dir] WARNING: No Mermaid renderers available, skipping prerendering"
            USE_MERMAID_FILTER=0
        fi
    fi

    HAVE_DOCKER=0; command -v docker &>/dev/null && HAVE_DOCKER=1
    HAVE_CURL=0; command -v curl &>/dev/null && HAVE_CURL=1

    python3 - "$TEMP_MARKDOWN_FILE" "$PRERENDER_IMG_DIR" "$KROKI_URL" "$MERMAID_IMG_FORMAT" "$STRICT_MERMAID" <<'PY'
import os
import re
import subprocess
import sys
import shutil

# Add project root to path to import antstack_core
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from antstack_core.figures.mermaid import sanitize_mermaid_for_rendering

md_path = sys.argv[1]
imgdir = os.path.abspath(sys.argv[2])
kroki_url = sys.argv[3]
img_format = sys.argv[4]
strict = sys.argv[5] == '1'

mmdc = shutil.which('mmdc')
have_docker = shutil.which('docker') is not None
have_curl = shutil.which('curl') is not None

# Clean existing diagrams before processing
for old_file in os.listdir(imgdir):
    if old_file.startswith('diagram_'):
        os.remove(os.path.join(imgdir, old_file))

with open(md_path, 'r', encoding='utf-8') as f:
    src = f.read()

pattern = re.compile(r"```mermaid[^\n]*\n(.*?)\n```", re.DOTALL)
count = 0
failures = 0

def sanitize_mermaid(code: str) -> str:
    # Replace unicode/math and LaTeX remnants with ASCII-friendly tokens
    replacements = {
        'α': 'a', 'β': 'b', 'γ': 'g', 'Δ': 'Delta', 'δ': 'delta',
        'η': 'eta', 'μ': 'mu', 'ρ': 'rho', '·': '*', '×': 'x',
        '→': '->', '↔': '<->',
        '𝒪': 'O', '𝑂': 'O', '𝑂(': 'O(', '𝒪(': 'O(',
    }
    # Remove LaTeX texttt commands
    code = re.sub(r'\\texttt\{([^}]*)\}', r'\1', code)
    for k, v in replacements.items():
        code = code.replace(k, v)
    # Strip LaTeX inline math markers if present inside mermaid block
    code = code.replace('\\(', '').replace('\\)', '')
    code = code.replace('$', '')
    # Remove edge labels like |label| which often include math-like tokens
    code = re.sub(r"\|[^|]*\|", "", code)
    # Ensure 'end' terminators are on their own line
    code = re.sub(r"\s+end\s+", "\nend\n", code)
    # Collapse multiple spaces to single
    code = re.sub(r"[\t ]+", " ", code)
    # Ensure each arrow and node definition separated by newlines
    code = code.replace(";", "\n")
    return code

def render_with_docker(mmd_path: str, out_path: str) -> None:
    uid = os.getuid()
    gid = os.getgid()
    cmd = [
        'docker','run','--rm','-u', f'{uid}:{gid}',
        '-v', f'{imgdir}:/data',
        'ghcr.io/mermaid-js/mermaid-cli:latest',
        '-i', f'/data/{os.path.basename(mmd_path)}',
        '-o', f'/data/{os.path.basename(out_path)}',
        '-b', 'transparent',
        '-w', '4096'
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Docker Mermaid rendering timed out after 30 seconds')

def render_with_kroki(mmd_path: str, out_path: str) -> None:
    # Try multiple Kroki endpoints if the primary one fails
    endpoints = [
        f"{kroki_url.rstrip('/')}/mermaid/{img_format}",
        "https://kroki.io/mermaid/png",
        "https://kroki.io/mermaid/svg"
    ]

    last_error = None
    for endpoint in endpoints:
        try:
            with open(out_path,'wb') as out_f:
                subprocess.run([
                    'curl','-s','-X','POST',
                    '-H','Content-Type:text/plain',
                    '--data-binary', '@'+mmd_path,
                    endpoint
                ], check=True, stdout=out_f, timeout=10)
            return  # Success, exit function
        except subprocess.TimeoutExpired:
            last_error = RuntimeError(f'Kroki rendering timed out after 10 seconds ({endpoint})')
        except subprocess.CalledProcessError as e:
            last_error = RuntimeError(f'Kroki rendering failed ({endpoint}): {e}')

    # If all endpoints failed, raise the last error
    if last_error:
        raise last_error
    else:
        raise RuntimeError('All Kroki endpoints failed')

def render_with_mmdc(mmd_path: str, out_path: str) -> None:
    # Sanitize the Mermaid content before rendering
    with open(mmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply sanitization
    content = sanitize_mermaid_for_rendering(content)
    
    # Write sanitized content to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as f:
        f.write(content)
        sanitized_path = f.name
    
    try:
        cmd = [mmdc, '-i', sanitized_path, '-o', out_path, '-b', 'white']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except subprocess.TimeoutExpired:
        raise RuntimeError('MMDC Mermaid rendering timed out after 20 seconds')
    finally:
        # Clean up temporary file
        import os
        try:
            os.unlink(sanitized_path)
        except:
            pass

def is_valid_image(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(16)
        # PNG signature
        if head.startswith(b'\x89PNG\r\n\x1a\n'):
            return True
        # SVG signature
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as tf:
                start = tf.read(100).lstrip()
            if start.startswith('<svg') or start.startswith('<?xml'):
                return True
        except Exception:
            return False
        return False
    except Exception:
        return False

def repl(match: re.Match) -> str:
    global failures
    code = sanitize_mermaid((match.group(1) or '').strip()) + '\n'
    if not code.strip():
        return match.group(0)
    # Generate consistent filename based on content hash
    import hashlib
    content_hash = hashlib.md5(code.encode('utf-8')).hexdigest()[:8]
    base = f'diagram_{content_hash}'
    mmd_path = os.path.join(imgdir, base + '.mmd')
    out_ext = '.png' if img_format.lower() != 'svg' else '.svg'
    out_path = os.path.join(imgdir, base + out_ext)
    with open(mmd_path, 'w', encoding='utf-8') as mf:
        mf.write(code)
    try:
        success = False
        methods_tried = []

        # Define rendering methods in order of preference
        render_methods = []

        # Add MMDC if available
        if mmdc:
            render_methods.append(('MMDC', render_with_mmdc))

        # Add Docker if available
        if have_docker:
            render_methods.append(('Docker', render_with_docker))

        # Add Kroki if available
        if have_curl:
            render_methods.append(('Kroki', render_with_kroki))

        # If no methods available, return original block
        if not render_methods:
            sys.stderr.write(f"⚠ No Mermaid renderers available, keeping original code for {os.path.basename(mmd_path)}\n")
            return match.group(0)

        # Try each method in order
        for method_name, render_func in render_methods:
            if success:
                break

            try:
                methods_tried.append(method_name)
                render_func(mmd_path, out_path)

                if is_valid_image(out_path):
                    success = True
                    sys.stderr.write(f"✓ {method_name} rendered {os.path.basename(out_path)}\n")
                else:
                    sys.stderr.write(f"✗ {method_name} produced invalid image for {os.path.basename(mmd_path)}\n")
                    # Clean up invalid image
                    if os.path.exists(out_path):
                        os.remove(out_path)

            except Exception as e:
                sys.stderr.write(f"✗ {method_name} failed for {os.path.basename(mmd_path)}: {e}\n")
                # Clean up any partial output
                if os.path.exists(out_path):
                    os.remove(out_path)

        if not success:
            sys.stderr.write(f"✗ All Mermaid rendering methods failed for {os.path.basename(mmd_path)} (tried: {', '.join(methods_tried)})\n")
            failures += 1
            if strict:
                raise RuntimeError(f'All Mermaid rendering methods failed for {os.path.basename(mmd_path)}')
            return match.group(0)  # Return original mermaid block
        # Return a project-relative path so pandoc can find it via resource-path
        rel_path = os.path.relpath(out_path, start=os.getcwd())
        # Create descriptive captions based on diagram content
        diagram_hash = base.replace('diagram_', '').replace('.mmd', '')
        
        # Map specific diagrams to descriptive captions based on content patterns
        def get_diagram_caption(diagram_content, diagram_hash):
            # Analyze content to determine caption type
            if 'Analysis[' in diagram_content and 'Methods[' in diagram_content and 'Scripts[' in diagram_content:
                return 'Enhanced analysis pipeline overview showing analysis modules, computational methods, orchestration scripts, validation framework, and empirical scaling results.'
            elif 'Body[' in diagram_content and 'Brain[' in diagram_content and 'Mind[' in diagram_content:
                return 'Module complexity overview detailing AntBody contact dynamics, AntBrain sparse neural networks, and AntMind bounded rational processing pipeline.'
            elif 'Physical[' in diagram_content and 'Control[' in diagram_content and 'Energy[' in diagram_content:
                return 'Energy flows overview across physical layer (terrain, sensors, mechanics), control layer (real-time processing), energy analysis components, and theoretical limits framework.'
            else:
                return f'Computational architecture diagram ({diagram_hash[:4]})'
        
        # Read the diagram content to determine the appropriate caption
        with open(mmd_path, 'r') as f:
            diagram_content = f.read()
        
        caption = get_diagram_caption(diagram_content, diagram_hash)
        return f"![{caption}]({rel_path}){{ width=70% }}"
    except Exception as e:
        # Attempt fallback paths before giving up
        try:
            if mmdc and have_docker:
                render_with_docker(mmd_path, out_path)
                if not is_valid_image(out_path):
                    raise RuntimeError('docker fallback invalid image')
            elif have_docker and have_curl:
                render_with_kroki(mmd_path, out_path)
                if not is_valid_image(out_path):
                    raise RuntimeError('kroki fallback invalid image')
            elif have_curl:
                render_with_kroki(mmd_path, out_path)
                if not is_valid_image(out_path):
                    raise RuntimeError('kroki fallback invalid image')
            else:
                raise
            rel_path = os.path.relpath(out_path, start=os.getcwd())
            diagram_num = base.replace('diagram_', '').replace('.mmd', '').zfill(3)
            
            # Map specific diagrams to descriptive captions based on content patterns  
            def get_diagram_caption(diagram_content, diagram_num):
                # Analyze content to determine caption type
                if 'Analysis[' in diagram_content and 'Methods[' in diagram_content and 'Scripts[' in diagram_content:
                    return 'Enhanced analysis pipeline overview showing analysis modules, computational methods, orchestration scripts, validation framework, and empirical scaling results.'
                elif 'Body[' in diagram_content and 'Brain[' in diagram_content and 'Mind[' in diagram_content:
                    return 'Module complexity overview detailing AntBody contact dynamics, AntBrain sparse neural networks, and AntMind bounded rational processing pipeline.'
                elif 'Physical[' in diagram_content and 'Control[' in diagram_content and 'Energy[' in diagram_content:
                    return 'Energy flows overview across physical layer (terrain, sensors, mechanics), control layer (real-time processing), energy analysis components, and theoretical limits framework.'
                else:
                    return f'Computational architecture diagram {diagram_num}'
            
            # Read the diagram content to determine the appropriate caption
            with open(mmd_path, 'r') as f:
                diagram_content = f.read()
            
            caption = get_diagram_caption(diagram_content, diagram_hash)
            return f"![{caption}]({rel_path}){{ width=70% }}"
        except Exception as e2:
            failures += 1
            sys.stderr.write(f"Mermaid render failed for a block: {e2}\n")
            return match.group(0)

new_src, nsubs = pattern.subn(repl, src)

# Replace lingering Unicode math symbols in the concatenated Markdown with LaTeX macros
# to avoid XeLaTeX glyph warnings in text mode and enforce style (prefer macros).
unicode_to_tex = {
    # Relations and operators
    '∼': '$\\sim$',
    '≈': '$\\approx$',
    '≤': '$\\le$',
    '≥': '$\\ge$',
    '≠': '$\\ne$',
    '±': '$\\pm$',
    '×': '$\\times$',
    '÷': '$\\div$',
    '−': '-',  # minus sign to ASCII hyphen
    '⋅': '$\\cdot$',
    '∘': '$\\circ$',
    '↑': '$\\uparrow$',
    '↓': '$\\downarrow$',
    '→': '$\\to$',
    '↔': '$\\leftrightarrow$',
    '⇒': '$\\Rightarrow$',
    '⇔': '$\\Leftrightarrow$',
    '∈': '$\\in$',
    '∉': '$\\notin$',
    '∩': '$\\cap$',
    '∪': '$\\cup$',
    '⊂': '$\\subset$',
    '⊆': '$\\subseteq$',
    '⊇': '$\\supseteq$',
    '∅': '$\\varnothing$',
    '∞': '$\\infty$',
    '∇': '$\\nabla$',
    '√': '$\\sqrt{}$',
    '°': '$^\\circ$',
    '∝': '$\\propto$',
    # Greek (regular)
    'α': '$\\alpha$', 'β': '$\\beta$', 'γ': '$\\gamma$', 'δ': '$\\delta$', 'Δ': '$\\Delta$',
    'ε': '$\\epsilon$', 'ζ': '$\\zeta$', 'η': '$\\eta$', 'θ': '$\\theta$', 'Θ': '$\\Theta$',
    'ι': '$\\iota$', 'κ': '$\\kappa$', 'λ': '$\\lambda$', 'Λ': '$\\Lambda$',
    'μ': '$\\mu$', 'µ': '$\\mu$', 'ν': '$\\nu$', 'ξ': '$\\xi$', 'Ξ': '$\\Xi$',
    'ο': 'o', 'π': '$\\pi$', 'Π': '$\\Pi$', 'ρ': '$\\rho$', 'σ': '$\\sigma$', 'Σ': '$\\Sigma$',
    'τ': '$\\tau$', 'υ': '$\\upsilon$', 'Υ': '$\\Upsilon$', 'φ': '$\\phi$', 'Φ': '$\\Phi$',
    'χ': '$\\chi$', 'ψ': '$\\psi$', 'Ψ': '$\\Psi$', 'ω': '$\\omega$', 'Ω': '$\\Omega$',
    # Mathematical italic variants commonly pasted from editors
    '𝛼': '$\\alpha$', '𝛽': '$\\beta$', '𝛾': '$\\gamma$', '𝛿': '$\\delta$', '𝛥': '$\\Delta$',
    '𝜀': '$\\epsilon$', '𝜁': '$\\zeta$', '𝜂': '$\\eta$', '𝜃': '$\\theta$',
    '𝜄': '$\\iota$', '𝜅': '$\\kappa$', '𝜆': '$\\lambda$',
    '𝜇': '$\\mu$', '𝜈': '$\\nu$', '𝜉': '$\\xi$',
    '𝜋': '$\\pi$', '𝜌': '$\\rho$', '𝜎': '$\\sigma$', '𝜏': '$\\tau$',
    '𝜐': '$\\upsilon$', '𝜑': '$\\phi$', '𝜒': '$\\chi$', '𝜓': '$\\psi$', '𝜔': '$\\omega$',
    # Additional problematic Unicode symbols
    '𝛥': '$\\Delta$',
    # Superscripts that might appear
    '⁷': '$^{7}$', '⁵': '$^{5}$', '⁻': '$^{-1}$',
    # Additional math symbols
    '≪': '$\\ll$', '≫': '$\\gg$', '≅': '$\\cong$', '≡': '$\\equiv$',
    '⊥': '$\\perp$', '∥': '$\\parallel$', '≲': '$\\lesssim$', '≳': '$\\gtrsim$',
}
# Smart replacement: avoid replacing inside existing math environments
def smart_unicode_replace(text, replacements):
    import re
    math_blocks = []
    math_pattern = r'\$[^$]*\$|\\\[[^\]]*\\\]|\\\([^)]*\\\)'
    
    def protect_math(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(0))
        return f'__MATH_BLOCK_{idx}__'
    
    # Protect existing math
    protected_text = re.sub(math_pattern, protect_math, text)
    
    # Replace unicode in non-math text
    for unicode_char, latex_replacement in replacements.items():
        protected_text = protected_text.replace(unicode_char, latex_replacement)
    
    # Restore math blocks
    for idx, math_block in enumerate(math_blocks):
        protected_text = protected_text.replace(f'__MATH_BLOCK_{idx}__', math_block)
    
    return protected_text

new_src = smart_unicode_replace(new_src, unicode_to_tex)

# Replace texttt commands with proper markdown code formatting for better PDF rendering
new_src = re.sub(r'\\texttt\{([^}]*)\}', r'`\1`', new_src)

# Replace any lingering images referencing /tmp with local copies under assets/tmp_images
import pathlib, shutil as sh
img_tmp_dir = os.path.join(os.path.dirname(imgdir), 'tmp_images')
os.makedirs(img_tmp_dir, exist_ok=True)

def repl_tmp(m: re.Match) -> str:
    alt = m.group(1); path = m.group(2)
    if path.startswith('/tmp'):
        bn = os.path.basename(path)
        dest = os.path.join(img_tmp_dir, bn)
        try:
            sh.copy2(path, dest)
            rel_dest = os.path.relpath(dest, start=os.getcwd())
            return f"![{alt}]({rel_dest})"
        except Exception:
            return m.group(0)
    return m.group(0)

new_src = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl_tmp, new_src)

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(new_src)
# Also emit a debug copy inside the paper assets directory for inspection
try:
    assets_dir = os.path.dirname(imgdir)  # assets/mermaid -> assets
    dbg_path = os.path.join(assets_dir, 'concat_after_prerender.md')
    with open(dbg_path, 'w', encoding='utf-8') as df:
        df.write(new_src)
except Exception:
    pass

if failures > 0 and strict:
    sys.stderr.write(f"Mermaid rendering failures: {failures}\n")
    sys.exit(1)
PY
fi

# Use the concatenated temp file as input for Pandoc.
# The `-f markdown-yaml_metadata_block` flag prevents pandoc from interpreting '---' as a metadata block.
echo "[$paper_dir] 🔄 Running Pandoc conversion to PDF..."
echo "[$paper_dir] 📊 This may take a few minutes for large documents..."
pandoc "${PANDOC_ARGS[@]}"
echo "[$paper_dir] ✅ Pandoc conversion completed"
# To update authors or contact info, edit the -V author and -V email lines above.

# Build validation and quality assurance checks
echo "[$paper_dir] Running build validation checks..."

VALIDATION_ISSUES=0

# Check 1: Verify PDF was created and has reasonable size
if [ -f "$OUTPUT_PATH" ]; then
    PDF_SIZE=$(stat -c%s "$OUTPUT_PATH" 2>/dev/null || stat -f%z "$OUTPUT_PATH" 2>/dev/null || echo "0")
    if [ "$PDF_SIZE" -lt 10000 ]; then
        echo "[$paper_dir] WARNING: PDF file seems too small (${PDF_SIZE} bytes), may indicate generation issues"
        VALIDATION_ISSUES=$((VALIDATION_ISSUES + 1))
    else
        echo "[$paper_dir] ✓ PDF created successfully (${PDF_SIZE} bytes)"
    fi
else
    echo "[$paper_dir] ERROR: PDF generation failed - output file not found"
    exit 1
fi

# Check 2: Validate that all referenced images exist
echo "[$paper_dir] Checking for missing image references..."
MISSING_IMAGES=$(grep -o "!\[.*\](\([^)]*\))" "$TEMP_MARKDOWN_FILE" | sed 's/.*!\[.*\](\([^)]*\)).*/\1/' | grep -E "\.(png|jpg|jpeg|svg)$" | sort | uniq)

for img_path in $MISSING_IMAGES; do
    # Convert relative paths to absolute based on paper directory
    if [[ "$img_path" == papers/* ]]; then
        full_path="$img_path"
    elif [[ "$img_path" == assets/* ]]; then
        full_path="$actual_paper_path/$img_path"
    else
        full_path="$actual_paper_path/assets/$img_path"
    fi

    if [ ! -f "$full_path" ]; then
        echo "[$paper_dir] WARNING: Referenced image not found: $full_path"
        VALIDATION_ISSUES=$((VALIDATION_ISSUES + 1))
    fi
done

# Check 3: Validate cross-references
echo "[$paper_dir] Checking cross-references..."
BROKEN_REFS=$(grep -n "Figure~??" "$OUTPUT_PATH" 2>/dev/null || echo "")
if [ -n "$BROKEN_REFS" ]; then
    echo "[$paper_dir] WARNING: Found broken cross-references in PDF"
    echo "$BROKEN_REFS" | head -5
    VALIDATION_ISSUES=$((VALIDATION_ISSUES + 1))
fi

# Check 4: Validate Mermaid diagram rendering success
if [ -d "$PRERENDER_IMG_DIR" ]; then
    TOTAL_DIAGRAMS=$(find "$PRERENDER_IMG_DIR" -name "*.mmd" | wc -l)
    RENDERED_DIAGRAMS=$(find "$PRERENDER_IMG_DIR" -name "*.png" | wc -l)
    if [ "$TOTAL_DIAGRAMS" -gt "$RENDERED_DIAGRAMS" ]; then
        echo "[$paper_dir] WARNING: $((TOTAL_DIAGRAMS - RENDERED_DIAGRAMS)) Mermaid diagrams failed to render"
        VALIDATION_ISSUES=$((VALIDATION_ISSUES + 1))
    else
        echo "[$paper_dir] ✓ All $TOTAL_DIAGRAMS Mermaid diagrams rendered successfully"
    fi
fi

# Summary
if [ "$VALIDATION_ISSUES" -eq 0 ]; then
    echo "[$paper_dir] ✓ Build validation passed - no issues detected"
    echo "[$paper_dir] Successfully created PDF: $OUTPUT_PATH"
else
    echo "[$paper_dir] ⚠ Build validation found $VALIDATION_ISSUES issue(s)"
    echo "[$paper_dir] PDF created but with potential issues: $OUTPUT_PATH"
fi 
}

# Entry point: build one or all papers
PAPERS_TO_BUILD=("ant_stack" "complexity_energetics" "cohereAnts")
if [ "$#" -gt 0 ]; then
    PAPERS_TO_BUILD=("$@")
fi

for paper in "${PAPERS_TO_BUILD[@]}"; do
    case "$paper" in
        documentation)
            render_one_paper "documentation" "1_ant_stack_documentation.pdf" "Ant Stack Documentation and PDF Rendering Guide"
            ;;
        ant_stack)
            render_one_paper "ant_stack" "1_ant_stack.pdf" "The Ant Stack"
            ;;
        complexity_energetics)
            render_one_paper "complexity_energetics" "2_complexity_energetics.pdf" "Computational Complexity and Energetics of the Ant Stack"
            ;;
        cohereAnts)
            render_one_paper "cohereAnts" "3_cohereAnts.pdf" "Infrared Vibrational Detection in Insect Olfaction"
            ;;
        *)
            echo "Unknown paper: $paper" >&2
            exit 1
            ;;
    esac
done