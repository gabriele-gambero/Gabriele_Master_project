import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

list_files(".")


#%%
## FONTS
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_dirs = ["../fonts/"]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


import regex
# List all available font names
available_fonts = sorted([f.name for f in font_manager.fontManager.ttflist])
print(available_fonts)


# Let's check:
print(plt.rcParams['font.family'])  # Should output 'YourDesiredFontName'

# Test plot to check the font
plt.plot([0, 1], [0, 1])
plt.title("Testing Font Change")
plt.show()

#%%


