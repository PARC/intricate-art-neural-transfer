This repository contains code and example images to generate intricate art designs. The corresponding blog article that goes along with this code is found here:

In order to run the python program, the following arguments are required:

- content image (found in silhouettes folder)
- style image (found in style folder)
- path to save the output 

Example use case is,

python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern

The default number of iterations is 10. Optionally, this can be altered with --num_iter argument. 

python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern --num_iter 100

The code can also optionally add background color or image to the generated art. To add background color, the hex color code is specified as an optional argument:

python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern --background_color "#f4c242"

To add background image, the path to the image is given as an argument:

python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern --background_image background/star.jpg

Both background color and image cannot be given at the same time. If background image is given, it takes precedent.