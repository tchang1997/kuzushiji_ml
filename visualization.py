from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

fontsize = 100

font = ImageFont.truetype('../NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
df_train = pd.read_csv('../train.csv')
unicode_map = {codept: char for codept, char in pd.read_csv('../unicode_translation.csv').values}



def visualize_training_data(image_file, labels):
    labels = np.array(labels.split(' ')).reshape(-1, 5)
    src_img = Image.open(image_file).convert('RGBA')
    bbox_canvas = Image.new('RGBA', src_img.size)
    char_canvas = Image.new('RGBA', src_img.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = unicode_map[codepoint]
        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255), width=3)
        char_draw.text((x+w+fontsize/4, y+h/2-fontsize), char, fill=(0, 0, 255, 255), font=font)

    src_img = Image.alpha_composite(Image.alpha_composite(src_img, bbox_canvas), char_canvas).convert("RGB")
    return np.asarray(src_img)

if __name__ == '__main__':
    filepath = sys.argv[1]
    shortname = filepath.split('/')[-1].split('.')[0]
    img_info = df_train[df_train['image_id'].str.contains(shortname)]
    plt.figure(figsize=(15, 15))
    plt.title(''.join(img_info['image_id']))
    plt.imshow(visualize_training_data(filepath, ''.join(img_info['labels'])), interpolation='lanczos')
    plt.show()
