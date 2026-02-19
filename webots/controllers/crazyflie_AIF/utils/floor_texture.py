from PIL import Image, ImageDraw

# -----------------------------
# Grid definition
# -----------------------------

NUM_COLS = 11   # horizontal cells
NUM_ROWS = 13   # vertical cells

# -----------------------------
# Texture resolution (square, power of 2)
# -----------------------------

TEXTURE_SIZE = 4096  # 4096 x 4096

# -----------------------------
# Appearance
# -----------------------------

BACKGROUND_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
LINE_WIDTH = 5

OUTPUT_FILENAME = "floor_grid_11x13_square.png"

# -----------------------------
# Image creation
# -----------------------------

img = Image.new("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), BACKGROUND_COLOR)
draw = ImageDraw.Draw(img)

# Compute cell size (must be square)
cell_size = min(
    TEXTURE_SIZE / NUM_COLS,
    TEXTURE_SIZE / NUM_ROWS
)

grid_width = cell_size * NUM_COLS
grid_height = cell_size * NUM_ROWS

# Center the grid
offset_x = (TEXTURE_SIZE - grid_width) / 2
offset_y = (TEXTURE_SIZE - grid_height) / 2

# Vertical grid lines
for col in range(NUM_COLS + 1):
    x = round(offset_x + col * cell_size)
    draw.line(
        [(x, round(offset_y)), (x, round(offset_y + grid_height))],
        fill=LINE_COLOR,
        width=LINE_WIDTH
    )

# Horizontal grid lines
for row in range(NUM_ROWS + 1):
    y = round(offset_y + row * cell_size)
    draw.line(
        [(round(offset_x), y), (round(offset_x + grid_width), y)],
        fill=LINE_COLOR,
        width=LINE_WIDTH
    )

# Save image
img.save(OUTPUT_FILENAME, "PNG")

print(
    f"Saved '{OUTPUT_FILENAME}' ({TEXTURE_SIZE} x {TEXTURE_SIZE})\n"
    f"Grid: {NUM_COLS} × {NUM_ROWS} cells\n"
    f"Cell size: {cell_size:.2f} px\n"
    f"Grid area: {int(grid_width)} x {int(grid_height)} px"
)
