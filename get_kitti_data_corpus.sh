#!/bin/bash

KITTI_BASE_URL=<YOURBASEURL_HERE>

files=(data_object_image_2.zip
data_object_label_2.zip
data_object_velodyne.zip
data_object_calib.zip
raw_data_downloader.zip
devkit_object.zip)

for i in ${files[@]}; do

	echo "Downloading: "$i
        wget -c -t 0 -T 8 $KITTI_BASE_URL/$i
        unzip -o $i
        rm $i
done

mkdir raw_data_downloader
cd raw_data_downloader
../raw_data_downloader.sh
