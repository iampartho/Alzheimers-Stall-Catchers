import csv
import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd





src_directory = "../../dataset/micro/"
files = [f for f in glob.glob(src_directory + "*" + ".mp4", recursive=True)]
already_downloaded = []

for f in files:
    already_downloaded.append(f.replace(src_directory, ""))

train_labels_file = "../../dataset/csv_files/train_labels.csv"
with open(train_labels_file, mode='r') as infile:
    reader = csv.reader(infile)
    train_list = {rows[0]: rows[1] for rows in reader}
    infile.close()
train_list.pop("filename", None)

total_stalled = 0
total_nonstall = 0

for data in train_list.keys():
    if train_list[data] == '1':
        total_stalled = total_stalled + 1
    elif train_list[data] == '0':
        total_nonstall = total_nonstall + 1

print("Already available files: " + str(len(already_downloaded)))
print("Files in full dataset: " + str(len(train_list)) + " non stall: " +
      str(total_nonstall) + " stall: " + str(total_stalled))





# remove all the files that are already downloaded from the full list

for file in range(len(already_downloaded)):
    train_list.pop(already_downloaded[file], None)

unseen_stalled = []
unseen_nonstall = []

for data in train_list.keys():
    if train_list[data] == '1':
        unseen_stalled.append(data)
    elif train_list[data] == '0':
        unseen_nonstall.append(data)

print("Unseen non stall: " + str(len(unseen_nonstall)) + " stall: " + str(len(unseen_stalled)))




train_metadata_file = "../../dataset/csv_files/train_metadata.csv"


with open(train_metadata_file, mode='r') as infile:
    reader = csv.reader(infile)
    metadata = {rows[0]: rows[1] for rows in reader}
    infile.close()
metadata.pop("filename", None)


"""
unseen_stall_download_names = []
for i in range(len(unseen_stalled)):
    data = unseen_stalled[i]
    data = data.replace(".mp4", "")

    unseen_stall_download_names.append(data)

df = pd.DataFrame(unseen_stall_download_names)
df.to_csv("stall.csv", index=False, header=False)
"""

"""
for i in range(len(unseen_nonstall)):
    unseen_nonstall[i] = unseen_nonstall[i].replace(".mp4", "")

with open("download_nonstall.csv", mode='r') as infile:
    reader = csv.reader(infile)
    appended = [rows[0] for rows in reader]
    infile.close()

for i in range(len(appended)):
    unseen_nonstall.remove(appended[i])
"""

indices = np.arange(len(unseen_nonstall))
np.random.seed(0)
np.random.shuffle(indices)

for batch in range(15):
    unseen_nonstall_download_names = []

    index = np.copy(indices[batch*1000:(batch+1)*1000])
    index = np.sort(index)

    for i in range(1000):

        data = unseen_nonstall[index[i]]
        data = data.replace(".mp4", "")

        unseen_nonstall_download_names.append(data)

    df = pd.DataFrame(unseen_nonstall_download_names)
    df.to_csv("nonstall_" + str(batch+1) + ".csv", index=False, header=False)



print("done")

