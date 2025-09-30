--[[
Lua filter to handle cross-references for Pandoc to LaTeX conversion.
Converts Pandoc's {#fig:...} syntax to LaTeX \label{fig:...} commands.
]]

function Header(el)
  -- Handle figure headers with {#fig:...} attributes
  if el.identifier and el.identifier:match("^fig:") then
    -- Add LaTeX label after the header
    local label = pandoc.RawBlock('latex', '\\label{' .. el.identifier .. '}')
    return {el, label}
  end
  
  -- Handle table headers with {#tab:...} attributes  
  if el.identifier and el.identifier:match("^tab:") then
    local label = pandoc.RawBlock('latex', '\\label{' .. el.identifier .. '}')
    return {el, label}
  end
  
  -- Handle section headers with {#sec:...} attributes
  if el.identifier and el.identifier:match("^sec:") then
    local label = pandoc.RawBlock('latex', '\\label{' .. el.identifier .. '}')
    return {el, label}
  end
  
  -- Handle equation headers with {#eq:...} attributes
  if el.identifier and el.identifier:match("^eq:") then
    local label = pandoc.RawBlock('latex', '\\label{' .. el.identifier .. '}')
    return {el, label}
  end
  
  return el
end

-- Also ensure that display math with labels gets proper LaTeX labels
function Math(el)
  if el.mathtype == "DisplayMath" then
    -- Check if there's a label in the content
    local label_pattern = "\\label%{([^}]+)%}"
    local label = el.text:match(label_pattern)
    if not label then
      -- Check for \tag or other label patterns
      return el
    end
  end
  return el
end

return {
  {Header = Header},
  {Math = Math}
}
