# VFX 2017 hw1 - High Dynamic Range Imaging
B09301056 孫凡耕, B03901119, 陳尚緯

## Followings should be installed and versions suggested:
- gcc/g++ 6.3.1
- **Opencv 3.2**
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
-h [ --help ]                               Print help message.
                                            "arg" implies that there should be an adequate amout of
                                            argument in that position.
                                            Values in the parentheses shows the default value if the user
                                            omit the argument.
-i [ --in_dir ] arg (=input_image)          Input directory (all images and input.txt should be under it).
                                            More explanation about the format of input.txt will be
                                            explained later.
-o [ --out_hdr_file ] arg (=result/out.hdr) Output filename of hdr (including .hdr).
-j [ --out_jpg_file ] arg (=result/out.jpg) Output filename of jpg (including .jpg).
-a [ --align ] arg                          Align images before processing using median threshold bitmap.
                                            Default value of maximum level and denoise margin is 7 and 4. 
                                            To skip this step, set the maximum level to negative value.
-b [ --blob ] [=arg(=False)]                Add blob-removal using OpenCV's SimpleBlobDetector with 
                                            specifically fine tuned paramters. This means unless you 
                                            have found a good set of parameters for your pictures, or else
                                            blob-removal may worsen picture quality.
--blob_tune [=arg(=False)]                  Tune blob-removal parameters. Turning on this options will 
                                            pop out windows with trackbars so that user can tune various
                                            variables to see the blob-removal result almost immediately.
                                            However, the users should rememeber the parameters by
                                            themselves, because after tuning, the program will exit. Then,
                                            users are required to manually change the numbers in
                                            src/util.cpp/blob_removal. All parameters are changeable 
                                            including the for-loop condition or even remove the for-loop.
-g [ --ghost ] [=arg(=False)]               Add ghost-removal mask using EA Khan's method.
-s [ --spotlight ] arg                      The regions of interest of the first image to be enhance.
                                            Type 4*n numbers to represent all rectangls in the format:
                                            (x, y, w, h).

-v [ --verbose ] [=arg(=False)]             Show the final result. If False, image will be saved without
                                            popping out.
-m [ --method ] arg (=1)                    Method to produce high-dynamic range image:
                                              0: hdr
                                              1: exposure fusion

--hdr_type arg (=0)                         Type of hdr:
                                              0: Debevec
--hdr_para arg                              Parameters for the chosen hdr algorithm.
                                              0(Debevec): two parameters:
                                                weight of smoothness term(lambda) (default=5)
                                                number of sampled points (default=60)

--tonemape_type arg (=0)                    Type of tonemap:
                                              0:Reinhard
--tonemap_para arg                          Parameters for the chosen tonemapping algorithm.
                                              0(Reinhard): four paramters:
                                                intensity(f) (default=0)
                                                contrast(m) (default=0)
                                                light adaption(a) (default=1)
                                                chromatic adaption(c) (default=0)

--fusion_type arg (=0)                      Type of exposure fusion:
                                              0:Mertens
--fusion_para arg                           Parameters for the chosen exposure fusion algorithm.
                                              0(Mertens): three parameters:
                                                contrast(c) (default=1)
                                                saturation(s) (default=1)
                                                well-exposure(e) (default=1)
                                                maximum process level (default=8)
```
It should be noted that `method 0(hdr)` and `method 1(exposure fusion)` are two different approaches to
produce high-dynamic images. If `method 0` is chosen, beware of `hdr_para` and `tonemap_para`. On the
contrary, beware of `fusion_para` during `exposure fusion`.
### Make sure the followings are in the in_dir:
- A txt file named `input.txt` which looks like:
```
15
pic1.JPG 0.25
pic2.JPG 2
...
```
The first line contains a number N: the total number of pictures to be computed.
The following N lines show picture name and exposure time seperated by a space.
- All the pictures listed in input.txt (case sensitive). Beware that the first image listed will be the
  pivoted(fixed) one during the alignment.
