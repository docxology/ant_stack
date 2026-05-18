-- Lua filter to normalize stray Unicode math symbols to LaTeX macros in inline text
-- This is complementary to the Markdown pre-pass in render_pdf.sh; it works at AST level.

local replacements = {
  ['∼'] = '\\sim', ['≈'] = '\\approx', ['≤'] = '\\le', ['≥'] = '\\ge',
  ['≠'] = '\\ne', ['±'] = '\\pm', ['×'] = '\\times', ['÷'] = '\\div',
  ['⋅'] = '\\cdot', ['∘'] = '\\circ',
  ['→'] = '\\to', ['↔'] = '\\leftrightarrow', ['⇒'] = '\\Rightarrow', ['⇔'] = '\\Leftrightarrow',
  ['∈'] = '\\in', ['∉'] = '\\notin', ['∩'] = '\\cap', ['∪'] = '\\cup',
  ['⊂'] = '\\subset', ['⊆'] = '\\subseteq', ['⊇'] = '\\supseteq', ['∅'] = '\\varnothing',
  ['∞'] = '\\infty', ['∇'] = '\\nabla', ['°'] = '$^\\circ$',
  -- Greek (regular and italic variants)
  ['α'] = '\\alpha', ['β'] = '\\beta', ['γ'] = '\\gamma', ['δ'] = '\\delta', ['Δ'] = '\\Delta',
  ['ε'] = '\\epsilon', ['ζ'] = '\\zeta', ['η'] = '\\eta', ['θ'] = '\\theta', ['Θ'] = '\\Theta',
  ['ι'] = '\\iota', ['κ'] = '\\kappa', ['λ'] = '\\lambda', ['Λ'] = '\\Lambda',
  ['μ'] = '\\mu', ['µ'] = '\\mu', ['ν'] = '\\nu', ['ξ'] = '\\xi', ['Ξ'] = '\\Xi',
  ['π'] = '\\pi', ['Π'] = '\\Pi', ['ρ'] = '\\rho', ['σ'] = '\\sigma', ['Σ'] = '\\Sigma',
  ['τ'] = '\\tau', ['υ'] = '\\upsilon', ['Υ'] = '\\Upsilon', ['φ'] = '\\phi', ['Φ'] = '\\Phi',
  ['χ'] = '\\chi', ['ψ'] = '\\psi', ['Ψ'] = '\\Psi', ['ω'] = '\\omega', ['Ω'] = '\\Omega',
  -- Mathematical italic variants (common copy-paste)
  ['𝛼'] = '\\alpha', ['𝛽'] = '\\beta', ['𝛾'] = '\\gamma', ['𝛿'] = '\\delta', ['𝛥'] = '\\Delta',
  ['𝜀'] = '\\epsilon', ['𝜁'] = '\\zeta', ['𝜂'] = '\\eta', ['𝜃'] = '\\theta',
  ['𝜄'] = '\\iota', ['𝜅'] = '\\kappa', ['𝜆'] = '\\lambda',
  ['𝜇'] = '\\mu', ['𝜈'] = '\\nu', ['𝜉'] = '\\xi',
  ['𝜋'] = '\\pi', ['𝜌'] = '\\rho', ['𝜎'] = '\\sigma', ['𝜏'] = '\\tau',
  ['𝜐'] = '\\upsilon', ['𝜑'] = '\\phi', ['𝜒'] = '\\chi', ['𝜓'] = '\\psi', ['𝜔'] = '\\omega',
}

local function replace_unicode_with_tex(s)
  local changed = false
  for k, v in pairs(replacements) do
    if s:find(k, 1, true) then
      s = s:gsub(k, '$' .. v .. '$')
      changed = true
    end
  end
  return s, changed
end

function Str(el)
  local s, changed = replace_unicode_with_tex(el.text)
  if changed then return pandoc.RawInline('latex', s) end
  return nil
end

function Code(el)
  local s, changed = replace_unicode_with_tex(el.text)
  if changed then
    -- Preserve code styling while emitting LaTeX-safe text
    return pandoc.RawInline('latex', s)
  end
  return nil
end

function CodeBlock(el)
  local s, changed = replace_unicode_with_tex(el.text)
  if changed then
    return pandoc.RawBlock('latex', s)
  end
  return nil
end

function Math(el)
  -- leave math intact; handled by LaTeX
  return nil
end

