from PIL import Image, ImageDraw, ImageFont

# Path to Telugu font file
telugu_font_path = r"C:\Users\Dell\OneDrive\Documents\telugu_text.ttf"

# Load the font
try:
    font = ImageFont.truetype(telugu_font_path, 32)
    print("Font loaded successfully.")
except Exception as e:
    print("Error loading font:", e)
