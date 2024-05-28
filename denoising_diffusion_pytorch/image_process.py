import numpy as np
from PIL import Image

room_label = [(0, 'LivingRoom', 1, "PublicArea",[220, 213, 205]),
              (1, 'MasterRoom', 0, "Bedroom",[138, 113, 91]),
              (2, 'Kitchen', 1, "FunctionArea",[244, 245, 247]),
              (3, 'Bathroom', 0, "FunctionArea",[224, 225, 227]),
              (4, 'DiningRoom', 1, "FunctionArea",[200, 193, 185]),
              (5, 'ChildRoom', 0, "Bedroom",[198, 173, 151]),
              (6, 'StudyRoom', 0, "Bedroom",[178, 153, 131]),
              (7, 'SecondRoom', 0, "Bedroom",[158, 133, 111]),
              (8, 'GuestRoom', 0, "Bedroom",[189, 172, 146]),
              (9, 'Balcony', 1, "PublicArea",[244, 237, 224]),
              (10, 'Entrance', 1, "PublicArea",[238, 235, 230]),
              (11, 'Storage', 0, "PublicArea",[226, 220, 206]),
              (12, 'Wall-in', 0, "PublicArea",[226, 220, 206]),
              (13, 'External', 0, "External",[255, 255, 255]),
              (14, 'ExteriorWall', 0, "ExteriorWall",[0, 0, 0]),
              (15, 'FrontDoor', 0, "FrontDoor",[255,255,0]),
              (16, 'InteriorWall', 0, "InteriorWall",[128,128,128]),
              (17, 'InteriorDoor', 0, "InteriorDoor",[255,255,255])]

def get_color_map():
    color = np.array([
        [244,242,229], # living room
        [253,244,171], # bedroom
        [234,216,214], # kitchen
        [205,233,252], # bathroom
        [208,216,135], # balcony
        [249,222,189], # Storage
        [ 79, 79, 79], # exterior wall
        [255,225, 25], # FrontDoor
        [128,128,128], # interior wall
        [255,255,255]
    ],dtype=np.int64)
    cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
    return color[cIdx]
cmap = get_color_map()/255.0

def convert_gray_to_rgb(img):
    img = img.astype(np.int64)
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(18):
        rgb[img == i] = cmap[i]*255
    return rgb

def load_image(path):
    img = Image.open(path)
    img = np.array(img)
    rgb=convert_gray_to_rgb(img)
    rgb=Image.fromarray(rgb)
    rgb.save("0_rgb.png")
    return img

# load_image("data/dataset/output64/0.png")