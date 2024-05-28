from shutil import move
import os
import pandas as pd
from tqdm import tqdm

ref_path="./data/new/annotated_imgs"
test_list=os.listdir(ref_path)
old_image_path="./data/new/image"
old_mask_path="./data/new/mask"
old_text_path="./data/new/text"
new_image_path="./data/new/image_test"
new_mask_path="./data/new/mask_test"
new_text_path="./data/new/text_test"
a=pd.read_csv(old_text_path+"/rooms.csv")
for i in tqdm(test_list):
    try:
        move(old_image_path+"/"+i, new_image_path+"/"+i)
        move(old_mask_path+"/"+i.replace(".png","_mask.png"), new_mask_path+"/"+i.replace(".png","_mask.png"))
    except:
        pass
json_test_list=["./data/"+i for i in test_list]
a[a['0'].isin(json_test_list)].to_csv(new_text_path+"/rooms2.csv")
a[~a['0'].isin(json_test_list)].to_csv(old_text_path+"/rooms2.csv")


    