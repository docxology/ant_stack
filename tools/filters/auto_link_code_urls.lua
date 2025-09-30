-- Lua filter to transform code spans that contain bare URLs into clickable links
-- Example: `https://example.com` -> clickable link

local url_pattern = "^https?://[%w%p]-$"

function Code(el)
  local txt = pandoc.utils.stringify(el.text or el.c or "")
  if txt:match(url_pattern) then
    return pandoc.Link(txt, txt)
  end
  return nil
end


