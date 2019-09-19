from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

fontsize = 100

font = ImageFont.truetype('../NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
df_train = pd.read_csv('../train.csv')
unicode_map = {codept: char for codept, char in pd.read_csv('../unicode_translation.csv').values}

def display_best_anchor_mapping(image_file, gt_array, anchor_map_array, C):
    num_bboxes = gt_array.shape[0]
    assert num_bboxes == anchor_map_array.shape[0]

    src_img = Image.open("../input/" + image_file + ".jpg").convert('RGBA')
    gt_canvas = Image.new('RGBA', src_img.size)
    anchor_canvas = Image.new('RGBA', src_img.size)
    gt_draw = ImageDraw.Draw(gt_canvas)
    anchor_draw = ImageDraw.Draw(anchor_canvas)

    resized_img_width, resized_img_height = C._img_size[:2]
    fmap_output_width, fmap_output_height = resized_img_width // 16, resized_img_height // 16
    for bbox_index in range(num_bboxes):
        gt_x1 = int(gt_array[bbox_index, 0] * src_img.size[0] / C._img_size[0])
        gt_y1 = int(gt_array[bbox_index, 2] * src_img.size[1] / C._img_size[1])
        gt_x2 = int(gt_array[bbox_index, 3] * src_img.size[0] / C._img_size[0])
        gt_y2 = int(gt_array[bbox_index, 1] * src_img.size[1] / C._img_size[1])
        gt_draw.rectangle((gt_x1, gt_y1, gt_x2, gt_y2), fill=(255, 0, 0, 0), outline=(255, 0, 0, 255), width=4)
        
        jy = anchor_map_array[bbox_index, 0]
        ix = anchor_map_array[bbox_index, 1]
        anchor_dim = C._anchor_box_scales[anchor_map_array[bbox_index, 2]]
        anchor_aspect_ratio = C._anchor_box_ratios[anchor_map_array[bbox_index, 3]]
        anchor_width = int(anchor_dim * anchor_aspect_ratio[0] * src_img.size[0] / resized_img_width) 
        anchor_height = int(anchor_dim * anchor_aspect_ratio[1] * src_img.size[1] / resized_img_height)
        anc_x1 = int(src_img.size[0] / fmap_output_width * (ix + 0.5) - anchor_width / 2) 
        anc_x2 = int(src_img.size[0] / fmap_output_width * (ix + 0.5) + anchor_width / 2) 
        anc_y1 = int(src_img.size[1] / fmap_output_height * (jy + 0.5) - anchor_height / 2) 
        anc_y2 = int(src_img.size[1] / fmap_output_height * (jy + 0.5) + anchor_height / 2)
        anchor_draw.rectangle((anc_x1, anc_y1, anc_x2, anc_y2), fill=(0, 255, 0, 0), outline=(0, 255, 0, 255), width=4)
    src_img = Image.alpha_composite(src_img, gt_canvas)
    src_img = Image.alpha_composite(src_img, anchor_canvas)
    plt.figure(figsize=(15, 15))
    plt.title("Ground-Truth Boxes with Optimal Anchor Mapping ({})".format(image_file))
    plt.imshow(np.asarray(src_img), interpolation='lanczos')
    plt.show()

def project_anchors(image_file, img_dimension_calc_fn, C):
    src_img = Image.open("../input/" + image_file + ".jpg").convert('RGBA')
    anchor_canvases = [Image.new('RGBA', src_img.size)] * 3
    anchor_draws = [ImageDraw.Draw(anchor_canvas) for anchor_canvas in anchor_canvases]
    resized_img_width, resized_img_height = C._img_size[:2]
    fmap_output_width, fmap_output_height = img_dimension_calc_fn(resized_img_width, resized_img_height) 
    anchor_colors = [(255, 0, 0, 10), (0, 255, 0, 10), (0, 0, 255, 10)]
    # for each anchor shape...
    anchors_calc = 0
    for anchor_area_index, anchor_dim in enumerate(C._anchor_box_scales):
        anchor_color = anchor_colors[anchor_area_index]
        anchor_outline_color = anchor_color[:3] + (100,)
        if anchor_area_index == 2: break
        for anchor_ratio_index, anchor_aspect_ratio in enumerate(C._anchor_box_ratios):
            
            # calculate anchor dimensions
            anchor_width = int(anchor_dim * anchor_aspect_ratio[0] * src_img.size[0] / resized_img_width) 
            anchor_height = int(anchor_dim * anchor_aspect_ratio[1] * src_img.size[1] / resized_img_height)

            for ix in range(fmap_output_width):
                # calculate anchor coordinates

                x_min = int(src_img.size[0] / fmap_output_width * (ix + 0.5) - anchor_width / 2) 
                x_max = int(src_img.size[0] / fmap_output_width * (ix + 0.5) + anchor_width / 2) 
                if x_min < 0 or x_max > src_img.size[0]: 
                    continue

                for jy in range(fmap_output_height): 
                    y_min = int(src_img.size[1] / fmap_output_height * (jy + 0.5) - anchor_height / 2) 
                    y_max = int(src_img.size[1] / fmap_output_height * (jy + 0.5) + anchor_height / 2)
                    if y_min < 0 or y_max > src_img.size[1]: # discard out of bounds 
                        continue

                    anchor_draws[anchor_area_index].rectangle((x_min, y_min, x_max, y_max), fill=anchor_color, outline=anchor_outline_color, width=4)

    
    src_img = Image.alpha_composite(src_img, anchor_canvases[0])
    src_img = Image.alpha_composite(src_img, anchor_canvases[1])
    # src_img = Image.alpha_composite(src_img, anchor_canvases[2])
    return np.asarray(src_img)
    


 

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
