from PIL import Image, ImageDraw

# -----------------------------
# Grid definition 
# -----------------------------

NUM_COLS = 11   # horizontal cells (x direction, width)
NUM_ROWS = 13   # vertical cells (y direction, height)

# -----------------------------
# Texture resolution
# -----------------------------
# Width and height are powers of two
# Aspect ratio = 11 : 13

TEXTURE_WIDTH = 4096
TEXTURE_HEIGHT = TEXTURE_WIDTH * NUM_ROWS // NUM_COLS  # keeps exact ratio

# -----------------------------
# Appearance
# -----------------------------

BACKGROUND_COLOR = (255, 255, 255)  # white
LINE_COLOR = (0, 0, 0)              # black
LINE_WIDTH = 5                      # thin lines (pixels)

OUTPUT_FILENAME = "floor_grid_11x13.jpg"

# -----------------------------
# Image creation
# -----------------------------

img = Image.new("RGB", (TEXTURE_WIDTH, TEXTURE_HEIGHT), BACKGROUND_COLOR)
draw = ImageDraw.Draw(img)

cell_width = TEXTURE_WIDTH / NUM_COLS
cell_height = TEXTURE_HEIGHT / NUM_ROWS

# Vertical grid lines (columns)
for col in range(NUM_COLS + 1):
    x = round(col * cell_width)
    draw.line([(x, 0), (x, TEXTURE_HEIGHT)], fill=LINE_COLOR, width=LINE_WIDTH)

# Horizontal grid lines (rows)
for row in range(NUM_ROWS + 1):
    y = round(row * cell_height)
    draw.line([(0, y), (TEXTURE_WIDTH, y)], fill=LINE_COLOR, width=LINE_WIDTH)

# Save image
img.save("floor_grid_texture.png", "PNG")

print(
    f"Saved '{OUTPUT_FILENAME}' "
    f"({TEXTURE_WIDTH} x {TEXTURE_HEIGHT}) "
    f"with {NUM_COLS} columns × {NUM_ROWS} rows"
)
