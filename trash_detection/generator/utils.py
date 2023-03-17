def get_bbox(left, upper, width, height):
  right = left+width
  lower = upper+height
  return left, upper, width, height