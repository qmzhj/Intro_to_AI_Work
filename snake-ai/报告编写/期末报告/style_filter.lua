-- style_filter.lua
function Para(el)
  return pandoc.Div(el, {['custom-style'] = 'HTML 预设格式'})
end
