#!/usr/bin/env python3
'''
Usage:
    flip_data.py --portion=<numeric_value> --path=<path>

Note:
    Augmented image and data from old tubs to
'''

from docopt import docopt
import cv2
import glob
import random
import json
import os
from progress.bar import IncrementalBar
import traceback


def flip_data(base_paths, portion):
    """
    Augmented image and data from old tubs to

    :param base_paths:          legacy tub paths
    :param porzion:             fraction of all data between [0,1]
    :return:                    None
    """
    if type(base_paths) is str:
        base_paths = [base_paths]
    
    for base_path in base_paths:
        #base_path = "C:\\Users\\Thema\\Documents\\projects\\mysim\\data\\lg_data\\circuit_launch_adam_1 - Copia"
        files_path = glob.glob(os.path.join(base_path,"record_*.json"))
        num_file = len(files_path)
        ite = int((len(files_path)-1)*float(portion))
        print("PATH: "+base_path)
        print("N° files: "+str(len(files_path)))
        print("AUGMENTED IMAGE: "+str(ite)+"/"+str(len(files_path)))
        bar = IncrementalBar('Augmenting', max=ite)
        while ite>0 and len(files_path)>0:
            try:
                index = random.randint(0, len(files_path)-1)
                
                with open(files_path[index]) as f:
                    data = json.load(f)
             
                
                if data["user/angle"] != 0:
                    # print(data)
                    #leggo immagine e la flip
                    src = cv2.imread(os.path.join(base_path,data["cam/image_array"]))
                    #cv2.imshow('windo2',src)
                    image = cv2.flip(src, 1)
                    #cv2.imshow('windo1',image)
                    # name = data["cam/image_array"].split('_')[1]
                    filename = str(num_file)+"_cam_image_array_.jpg"
                    # print("Filename: "+filename)
                    cv2.imwrite(os.path.join(base_path,filename), image)
                    data["user/angle"] = data["user/angle"]*-1
                    data["cam/image_array"] = filename
                    # name = f_name.split('.')[0]
                    json_name = 'record_'+str(num_file)+".json"                    
                    # print("Json_name: "+json_name)
                    with open(os.path.join(base_path, json_name), 'w') as outfile:
                        json.dump(data, outfile)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    ite = ite - 1
                    num_file = num_file+1
                    bar.next()
                files_path.remove(files_path[index])
            except Exception as exception:
                print(f'Error {base_path}\n', exception)
                traceback.print_exc()
            
        files_path = glob.glob(os.path.join(base_path,"record_*.json"))
        print(len(files_path))
        print("N° new file: "+str(len(files_path))+"\n")

    print("DONE")

if __name__ == '__main__':
    args = docopt(__doc__)
    portion = args["--portion"]
    input_path = args["--path"]
    paths = input_path.split(',')
    flip_data(paths, portion)