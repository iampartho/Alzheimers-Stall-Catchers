import csv
import os
from tqdm import tqdm


download_list = "nonstall_1.csv"

download_folder = "downloaded/"

start_pos = 0
end_pos = 10

with open(download_list, mode='r') as infile:
    reader = csv.reader(infile)
    files = [rows[0] for rows in reader]
    infile.close()


for i in tqdm(range(start_pos, end_pos)):
    data = files[i]
    url = "s3://drivendata-competition-clog-loss/train/" + data + ".mp4"
    os.system("aws s3 cp " + url + " " + download_folder + " --no-sign-request")

print("done")
