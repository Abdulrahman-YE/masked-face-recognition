

import argparse
from cmath import log
import os
from pathlib import Path
from typing import List
from person import Person, update_persons
from utils.utils import check_folder, load_img
from config import *
import numpy as np
from PIL import Image





    
def crop_face(img):
    face = FACE_DETECTOR( img)

    if len(face) == 1: 
        print("Face Found")
        x,y,w,h = face[0]
        return np.array(img[int(y):int(y+h),int(x):int(w)])
        
    else:
        return None
            
        
def get_features(img):
    
    tensor = NORMALIZE(TRANSFORM(img).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embed  = MASKED_MODEL(tensor)['embeddings']
        return embed.cpu().numpy().reshape(512)




def construct_person(img_file):
    name, img = load_img(img_file)
    name = name.split('.')[0]
    print(f'Name : {name}, size : {img.shape}')
    face = crop_face(img)
    if face is not None:
        return Person(name=name, embed=get_features(face) )

def add_persons(imgs_folder : Path):
    persons : List[Person] = []
    print(' =_=_=_=_=_=_=_=[Getting Persons]=_=_=_=_=_=_=_=')
    for i in os.listdir(imgs_folder):
        if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg'):
            person = construct_person(os.path.join(imgs_folder, i))
            if person is not None:
                persons.append(person)
        
    update_persons(persons)
    print(' =_=_=_=_=_=_=_=[DONE]=_=_=_=_=_=_=_=')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='images', help='المجلد الذي يحتوي على صور الاشخاص')

    opt = parser.parse_args()
    try:
        opt.srource = check_folder(opt.source)
    except Exception as e:
        log.error(f'An exception occurred while adding persons\nException : {e.__traceback__}')
    
    add_persons(Path(opt.source))
