# VFX 2017 hw1 - High Dynamic Range Imaging

## Followings should be installed and versions suggested:
- gcc/g++ 6.3.1
- opencv 3.2

## How to make:
Type `make` under `/hw1` directory and a `hdr` executable will be created.

## Usage
Execute as follows:
```
./hdr pics_dir/ mtb_max_level mtb_max_denoise debevec_lambda_value 
```
###Make sure the followings are in the pics_dir:
- A txt file named `input.txt` which looks like:
```
15
xxx.JPG 0.25
yyy.JPG 2
...
```
The first line contains a number N: the total number of pictures to be computed
The following N lines show picture name and exposure time seperated by a space.
- All the pictures listed in input.txt (case sensitive)
- Another txt file named `sample.txt` which listed all the sampled point in the
  following manner:
```
100
10 11
20 19
...
```
The first line contains a number N: the total sampled points.
The following N lines show the x and y coordinate seperated by a space.
###Descriptions of all hyper-parameters:
- mtb_max_level: The maximum level of image pyramid while aligning images using
  the median threshold bitmap (MTB).
- mtb_max_denoise: The threshold whenever a grayscaled pixel value is considered
  too close to the median, thus the pixel is included in the exclusion bitmap.
- debevec_lambda_value: The weight on the smoothness term for debevec hdr method.
  
