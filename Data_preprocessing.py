import numpy as np
import os
import fnmatch
import glob
from medpy.io import load
import cv2
from tqdm import tqdm
import time
import image_slicer
import pickle as pkl

file_path_save = "path to files names"
with open(file_path_save + "//" + "File Name", "rb") as fp:
  filenames = pkl.load(fp)

path_0 = "Path to CN files"
path_1 = "Path to AD files"
path_2 = "Path to MCI files"

''' Read CN, AD and MCI Paths
'''
file_list = []
for i in range(len(filenames[0])):
    if filenames[1][i] == 0:
        file_list.append(path_0 + "//" + filenames[0][i])
    elif filenames[1][i] == 1:
        file_list.append(path_1 + "//" + filenames[0][i])
    elif filenames[1][i] == 2:
        file_list.append(path_2 + "//" + filenames[0][i])

''' Read dcm files
'''
def data_load(path):
  file_path = []
  for root, folder, file in os.walk(os.path.abspath(path)):
    for filename in fnmatch.filter(file, "*.dcm" ):
          file_path.append(os.path.join(root, filename))

  return file_path

''' Sanity Check
'''
def check (all_files):
    check_filename = []
    for i in range(len(all_files)):
        check_filename.append(all_files[i].rsplit("\\")[9])
    print(set(check_filename))

all_files_ad = data_load(path_1)
check(all_files_ad)

all_files_cn = data_load(path_0)
check(all_files_cn)

all_files_mci = data_load(path_2)
check(all_files_mci)

def subject_id(files):
    ids = []
    for i in range(len(files)):
        ids.append(files[i].rsplit("\\")[-1].rsplit("_")[-1].rsplit(".")[0])

    return ids


ids_ad = list(set(subject_id(all_files_ad)))
ids_cn = list(set(subject_id(all_files_cn)))
ids_mci = list(set(subject_id(all_files_mci)))

''' Sanity check for Input
'''
def preprocess_step(indx_file, file_path):
    filepath = []
    for j in tqdm(indx_file):
        intermediate_filepath = []
        for k in range(len(file_path)):
            try:
                unique_id = file_path[k].rsplit("\\")[-1].rsplit("_")[-1].rsplit(".")[0]
            except:
                print("Check this file:", file_path[k])
                return 0

            if j == unique_id:
                intermediate_filepath.append(file_path[k])
        if len(intermediate_filepath) != 1:
            filepath.append(intermediate_filepath)
        time.sleep(0.5)
    return filepath

preprocess_filepath_ad = preprocess_step(ids_ad, all_files_ad)
preprocess_filepath_cn = preprocess_step(ids_cn, all_files_cn)
preprocess_filepath_mci = preprocess_step(ids_mci, all_files_mci)

files_ad2 = []
files_cn2 = []
files_mci2 = []

for j in range(len(filenames[0])):
    count_cn = 0
    count_ad = 0
    count_mci = 0
    for i in range(len(preprocess_filepath_ad)):
        idd = preprocess_filepath_ad[i][0].rsplit("\\")[-5]

        if filenames[0][j] == idd and count_ad == 0:
            files_ad2.append(preprocess_filepath_ad[i])
            count_ad += 1

    for i in range(len(preprocess_filepath_mci)):
        idd = preprocess_filepath_mci[i][0].rsplit("\\")[-5]

        if filenames[0][j] == idd and count_mci ==0:
            files_mci2.append(preprocess_filepath_mci[i])
            count_mci +=1

    for i in range(len(preprocess_filepath_mci)):
        idd = preprocess_filepath_mci[i][0].rsplit("\\")[-5]

        if filenames[0][j] == idd and count_cn ==0:
            files_mci2.append(preprocess_filepath_mci[i])
            count_cn +=1

''' Sort all data with Histogram Ranking Method
'''
def histo_preprocess(filepaths, size):
    len_a = len(filepaths)
    files = []
    for i in tqdm(range(len_a)):
        len_b = len(filepaths[i])

        counter = np.zeros(len_b)
        for j in range(len_b):
            img,_  = load(filepaths[i][j])
            img = img.astype(np.uint8)
            vals = img.reshape((img.shape[0],img.shape[1],1)).mean(axis=2).flatten()
            counts, bins = np.histogram(vals, range(257))
            counter[j] = counts[100:].sum()/counts.sum()
        counter_sort = np.argsort(counter)[-size-1:-1]

        intermediate_path = []
        for k in range(len(counter_sort)):
            intermediate_path.append(filepaths[i][counter_sort[k]])
        files.append(intermediate_path)

    return files

custom_size = 64
preprocess_filepath_ad1 = histo_preprocess(files_ad2, custom_size)
preprocess_filepath_cn1 = histo_preprocess(files_cn2, custom_size)
preprocess_filepath_mci1 = histo_preprocess(files_mci2, custom_size)

def listindxs(data):
    listindx = []
    for i in range(len(data)):
        ll = []
        for j in range(len(data[i])):
            ll.append(data[i][j].rsplit("_")[-3])
        listindx.append(ll)
    return listindx

list_ad = listindxs(preprocess_filepath_ad1)
list_cn = listindxs(preprocess_filepath_cn1)
list_mci = listindxs(preprocess_filepath_mci1)

def chn_to_int(lists):
    for i in range(len(lists)):
        for j in range(len(lists[i])):
            lists[i][j] = int(lists[i][j])
    return lists

list_ad = chn_to_int(list_ad)
list_cn = chn_to_int(list_cn)
list_mci = chn_to_int(list_mci)

''' Sort based on Index
'''
def sort_data(datapath, indxpath):
    final_data = []
    for i in range(len(datapath)):
        add_path = []
        sorted_list = np.argsort(np.array(indxpath[i]))
        for j in range(len(datapath[i])):
            add_path.append(datapath[i][sorted_list[j]])
        final_data.append(add_path)
    return final_data


preprocess_filepath_ad2 = sort_data(preprocess_filepath_ad1, list_ad)
preprocess_filepath_cn2 = sort_data(preprocess_filepath_cn1, list_cn)
preprocess_filepath_mci2 = sort_data(preprocess_filepath_mci1, list_mci)

#save file Preprocess_filepath_ad2
#save file Preprocess_filepath_cn2
#save file Preprocess_filepath_mci2