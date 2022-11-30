"""
Script to add the embeding of a person to json file
Structure of folder:
    Name_of_Person(folder):
    --> mask.[jpg][jpeg]
    --> no_mask.[jpg][jpeg]
"""

import argparse
from pathlib import Path
from typing import List
from person import Person, read_persons, update_persons
from utils.utils import load_img
from config import *
import numpy as np
from utils.logger import get_logger

# Set logger
log = get_logger("Test")


def crop_face(img) -> List:
    """
    Function to crop the face detected from img
    @params: 
        img : Image to detect face from.

    @returns:
        if face detected return np.array represnt the face otherwise None
    """
    face = FACE_DETECTOR(img)

    if len(face) == 1:
        print("Face Found")
        x, y, w, h = face[0]
        return np.array(img[int(y):int(y+h), int(x):int(w)])

    else:
        return None


def get_features(img):
    """
    Function to get features of masked face.
     @params: 
        img : Masked face to get features from.     
    @returns:
        np.array(512) represnt the features of the face.
    """
    tensor = NORMALIZE(TRANSFORM(img).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_embed = MASKED_MODEL(tensor)['embeddings']
        return mask_embed.cpu().numpy().reshape(512)


def get_face_features(img):
    """
    Function to get features of unmasked face.
     @params: 
        img : face to get features from.     
    @returns:
        np.array(512) represnt the features of the face.
    """
    tensor = NORMALIZE(TRANSFORM(img).float()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        face_embed = NORMAL_RESNET(tensor)
        return face_embed.cpu().detach().numpy().reshape(512)


if __name__ == '__main__':
    persons = list(read_persons())
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='images',
                        help='المجلد الذي يحتوي على صور الاشخاص')

    opt = parser.parse_args()

    source = Path(opt.source)
    person = Person(name=source.name)
    if source.exists():
        images = list(source.glob("*.[jpg][jpeg]*"))
        for i in images:
            img = load_img(i)
            face = crop_face(img)
            if face is not None:
                if "no_mask" in i.name:
                    person.no_mask_embed = get_face_features(face)
                elif "mask" in i.name:
                    person.mask_embed = get_features(face)
                else:
                    log.debug(f"Images must be labelled as no_mask")
            else:
                log.debug(f"Image {i.name} must only have 1 face in it.")

        if person.mask_embed is not None and person.no_mask_embed is not None:
            persons.append(person)
            update_persons(persons)
        else:
            if person.mask_embed is None:
                log.error(f"Embeding of masked face is null")
            if person.no_mask_embed is None:
                log.error(f"Embeding of no mask face is null")

    else:
        log.debug(f'Folder {source.absolute} does not exists.')

    # add_persons(Path(opt.source))
