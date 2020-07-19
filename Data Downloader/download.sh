counter=0
while read line; do

  counter=$((counter+1))
  echo "$counter: $line"

  if grep -Fxq "$line" ./completed.txt
  then
    # code if not found
    echo "Skipping $line"
  else
      echo "Downloading $line..."
      aws s3 cp s3://drivendata-competition-clog-loss/train/$line.mp4 train_videos/ --no-sign-request

      if [ $? -eq 0 ]; then
          echo "Successfully downloaded $line"
          echo $line >> ./completed.txt
      else
          echo "Failure in $line"
          break
      fi
  fi

# specify the target csv list of files to download
done <stall.csv
