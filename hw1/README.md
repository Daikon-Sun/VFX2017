# VFX 2017 hw1 - High Dynamic Range Imaging
by B09301056 孫凡耕, B03901119, 陳尚緯

## Followings should be installed and versions suggested:
- gcc/g++ 6.3.1
- Opencv 3.2
- Boost >= 1.5
- Cmake >= 3.0

## How to make:
Type `cmake .` under `VFX2017/hw1` directory and a `Makefile` will be created.
Then, type `make` and finally a `main` executable will be created.

## Usage
To see simple help message, simply type:
```
./main -h
```
Here is a detailed explanation of parameters:
```
-h [ --help ]                               Print help message. "arg" implies that there should be an
                                            adequate amout of argument in that position. Values in the
                                            parentheses shows the default value if the user omit the
                                            argument.
-i [ --in_dir ] arg (=input_image)          Input directory (all pictures and input.txt should be under it).
-o [ --out_hdr_file ] arg (=result/out.hdr) Output filename of hdr (including .hdr).
-j [ --out_jpg_file ] arg (=result/out.jpg) Output filename of jpg (including .jpg).
-a [ --align ] arg                          Align images before processing.
-g [ --ghost ] [=arg(=False)]               Add ghost-removal mask.
-v [ --verbose ] [=arg(=False)]             Show the final result.
-m [ --method ] arg (=1)                    Method to produce high-dynamic range 
                                            image:
                                              0: hdr
                                              1: exposure fusion
                                            
--hdr_type arg (=0)                         Type of hdr:
                                              0: Debevec
                                            
--hdr_para arg                              Parameters for the chosen hdr 
                                            algorithm.
--tonemape_type arg (=0)                    Type of tonemap:
                                              0:Reinhard
                                            
--tonemap_para arg                          Parameters for the chosen tonemapping 
                                            algorithm.
--fusion_type arg (=0)                      Type of exposure fusion:
                                            0:Mertens
                                            
--fusion_para arg                           Parameters for the chosen exposure 
                                            fusion algorithm.
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
  
