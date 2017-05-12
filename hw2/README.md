# VFX 2017 hw2 - Image Stitching
***B09301056 孫凡耕, B03901119 陳尚緯***

## Followings should be installed and versions suggested:
- gcc/g++ 6.3.1
- **Opencv 3.2**
- Boost >= 1.5
- Cmake >= 3.0
- Ceres-solver >= 1.12.0 (with Eigen >= 3.3)

## How to make:
Type `cmake .` under `VFX2017/hw2` directory and a `Makefile` will be created.
Then, type `make` and finally a `main` executable will be created.
To see our report, cd to `report` then type `make` then a `hw2.pdf` will be 
created.

## Usage
To see simple help message, simply type:
```
./main -h
```
Here is a detailed explanation of parameters:
```
All Available Options in VFX2017 hw2 project:
  -h [ --help ]                         Print help message.
  -i [ --in_list ] arg (=input_images.txt)
                                        List of all input images. (Image names 
                                        should be seperated by any spaces.)
  -o [ --out_prefix ] arg (=result/out) Output prefix of the panorama images 
                                        (Images for will be named as: 
                                        out_prefix0.jpg, out_prefix1.jpg...).
  -v [ --verbose ] [=arg(=True)]        Visualize the final result.
  -z [ --zoom ] arg (=0.25)             Scale the image according to this value
                                        before processing to achieve faster 
                                        result and use lesser memory. For 
                                        example: a [6000x4000] image with zoom 
                                        = 0.2 will become a [1200x800] image.
  -p [ --panorama ] arg (=1)            modes of generating panorama:
                                          0: linear ordering(n) (The program 
                                             will stitch the images according 
                                             to the order in "in_list" one 
                                             after one from left to right
                                          1: any ordering(n^2) (The program 
                                             will automatically stitch the 
                                             images in any order and recognise 
                                             all scenes to produce one panorama
                                             for a single scene
                                        
  -d [ --detection ] arg (=1)           modes of feature detection:
                                          0: MSOP (Multi-Scale Oriented Patches
                                          1: SIFT (Scale Invariant Feature 
                                             Transform
                                        
                                        
  -m [ --matching ] arg (=2)            modes of feature matching:
                                          0: exhaustive search
                                          1: HAAR wavelet-based hashing
                                          2: FLANN (Fast Library for 
                                             Approximate Nearest Neighbors
  --matching_para arg                   Parameters of the chosen feature 
                                        matching mode.
                                          0: (0, maximum y-coordinate 
                                        displacement for geometric constraint)
                                          1: (0, bin number), (1, threshold 
                                        between 1-NN / (avg 2-NN))
                                        (2, maximum y-coordinate displacement 
                                        for geometric constraint)
                                          2: NONE
                                        
                                        
  -j [ --projection ] arg (=1)          Types of projection:
                                          0: none
                                          1: cylindrical
  --projection_para arg                 Parameters of the chosen projection 
                                        type.
                                          0: NONE
                                          1: (0, focal length)
                                        
                                        
  -s [ --stitching ] arg (=4)           modes of image stitching:
                                          0: translation
                                          1: translation + estimate focal 
                                             length (deprecated)
                                          2: translation + rotation 
                                             (deprecated)
                                          3: homography
                                          4: automatic stitching
  --stitching_para arg                  Parameters of the chosen image 
                                        stitching mode.
                                          0: (0, number of rounds for RANSAC), 
                                        (1, threshold of inliner/outlier)
                                          3: (0, threshold of inliner/ouliear)
                                          4: (0, threshold of inliner/ouliear),
                                        (1, estimated focal length (nonzero) 
                                        for initialization of bundle 
                                        adjustment)
                                        
                                        
  -b [ --blending ] arg (=1)            modes of blending:
                                          0: average (simple average of 
                                             overlapping region)
                                          1: multi-band
  --blending_para arg                   Parameters of the chosen blending mode.
                                          0: NONE
                                          1: (0, number of bands)

```
For example, `./main -i input_images.txt -o result/out -z 0.25 -p 1 -d 1 -m 2
-j 1 --projection_para 2000 -s 4 --stitching_para 5 2000 --blending_para 4` will 
be a valid input. This is also the default value is you simple typed `./main`.
