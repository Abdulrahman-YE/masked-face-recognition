
from typing import Optional
import numpy as np
import orjson
from pydantic import BaseModel, validator
import argparse
from pathlib import Path
import os
from utils.logger import get_logger

log = get_logger("Person")
def orjson_dumps(v, *, default):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default, option=orjson.OPT_SERIALIZE_NUMPY).decode()

class Person(BaseModel):
    """
        كلاس يمثل البيانات التي يتم تخزينها لكل شخص
    """
    # اسم الشخص
    name : str
    # ملامح الشخص
    mask_embed : Optional[np.ndarray]
    no_mask_embed : Optional[np.ndarray]


    @validator('mask_embed', "no_mask_embed", pre=True)
    def toNumpy(cls, v):
        """
            دالة يتم استدعائها قبل ما يتم اسناد القيمة الى mask_embed.
            لتاكد من انها البيانات من نوع مصفوفة نمباي numpy
        """
        if isinstance(v, list):
            return np.array(v)
        return v
    
    class Config:
        arbitrary_types_allowed = True
        json_dumps = orjson_dumps
        json_loads = orjson.loads



def read_persons(file='persons.json'):
    """
        دالة تقراءبيانات الاشخاص من ملف جيسون و تحولها لمصوفة من كلاس الاشخاص
    """
    f =  open(file, 'r')
    persons =  *map(Person.parse_obj, orjson.loads(f.read())),
    f.close()
    return  persons




def update_persons(persons, file='persons.json'):
    """
        دالة تحدث بيانات الاشخاص من مصوفة  كلاس الاشخاص  الى ملف جيسون 
    """
    f =  open(file, 'w')
    f.write('[')
    num_of_persons = len(persons)
    for i, per in enumerate(persons):
        f.write(per.json())
        if i < num_of_persons -1:
            f.write(',')
    f.write(']')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='images', help='المجلد الذي يحتوي على صور الاشخاص')

    opt = parser.parse_args()
    if Path(opt.source).exists():
        source = Path(opt.source)
        for f in os.listdir(source):
            print(f)
            os.system(f"python test.py --source \"{source.joinpath(f)}\"")
    else:
        log.debug(f"Folder {opt.source} does not exists.")
    