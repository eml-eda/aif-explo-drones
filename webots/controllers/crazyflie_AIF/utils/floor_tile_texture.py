from PIL import Image, ImageDraw

SIZE = 512
LINE_WIDTH = 3

img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
draw = ImageDraw.Draw(img)

draw.rectangle(
    [0, 0, SIZE - 1, SIZE - 1],
    outline=(0, 0, 0),
    width=LINE_WIDTH
)

img.save("white_square_border_512.png", "PNG")
