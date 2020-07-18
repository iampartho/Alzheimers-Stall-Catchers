# Downloading More Data

1. First you need the awscli tool
   ```
   pip install awscli
   ```
   
2. Now you can run "downloader.py" python script.

   I have provided 17 different randomized lists for data download. 
   
   "stall.csv" contains all the stalled examples not included in the micro data.
   
   "nonstall_#.csv" contains subsets of 1000 non stalled data excluded from micro set.
   
   You can select the appropriate list from here:
   ```
   download_list = "nonstall_1.csv"
   ```
   Specify the download directory here:
   ```
   download_folder = "downloaded/"
   ```
   It takes quite some time to download the data. In case you think there might be network interruption, you can run the code multiple times by selecting a subset of the 1000 files listed in each csv file so that you can continue from checkpoints. 
   
   Use start_pos = 0 and end_pos = 1000 for all the files listed in a csv
   ```
   start_pos = 0 
   end_pos = 10
   ```

3. Ciao