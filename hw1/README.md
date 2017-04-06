# VFX 2017 hw1 - High Dynamic Range Imaging

## Followings should be installed and versions suggested:
- gcc/g++ 6.3.1
- opencv 3.2

## How to make:
Type `make` under `/hw1` directory and a `main` executable will be created.

## Usage
Execute as follows:
```
./main pics_dir/ mml mmd lambda hdr_filname jpg_filename
```
### Make sure the followings are in the pics_dir:
- A txt file named `input.txt` which looks like:
```
15
pic1.JPG 0.25
pic2.JPG 2
...
```
The first line contains a number N: the total number of pictures to be computed.
The following N lines show picture name and exposure time seperated by a space.
- All the pictures listed in input.txt (case sensitive).
### Descriptions of all hyper-parameters:
- mml (mtb_max_level): The maximum level of image pyramid while aligning images using
  the median threshold bitmap (MTB).
- mmd (mtb_max_denoise): The threshold whenever a grayscaled pixel value is considered
  too close to the median, thus the pixel is included in the exclusion bitmap.
- lambda: The weight on the smoothness term for debevec hdr method.
- hdr_filename: the output filename for hdr (remember to include ".hdr" at end).
- jpg_filename: the output filename for hdr (remember to include ".jpg" at end).
  
