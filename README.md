## Creating Intricate Art with Neural Style Transfer

This repository contains code and example images to generate intricate art designs. The corresponding blog article that goes along with this code is found here: https://medium.com/@kramea/creating-intricate-art-with-neural-style-transfer-e5fee5f89481

### Try it now

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run)

Click this button to open a Workspace on FloydHub to run this code.

### Details

In order to run the python program, the following arguments are required:

- content image (found in silhouettes folder)
- style image (found in style folder)
- path to save the output 

Example use case is,

```
python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern
```

The default number of iterations is 10. Optionally, this can be altered with --num_iter argument. 

```
python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern --num_iter 100
```

The code can also optionally add background color or image to the generated art. To add background color, the hex color code is specified as an optional argument:

```
python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern --background_color "#f4c242"
```

To add background image, the path to the image is given as an argument:

```
python intricate_style.py silhouettes/unicorn.jpg style/color_pattern3.jpg results/unicorn_color_pattern --background_image background/star.jpg
```

Both background color and image cannot be given at the same time. If background image is given, it takes precedent.


### References:

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. A Neural Algorithm of Artistic Style. 2015. https://arxiv.org/abs/1508.06576
- Roman Novak and Yaroslav Nikulin. Improving the Neural Algorithm of Artistic Style. 2016. https://arxiv.org/abs/1605.04603
- Some of the code is developed based on Somshubra Majumdar’s implementation of neural style transfer: https://github.com/titu1994/Neural-Style-Transfer
- Geometric patterns are obtained from artist Rebecca Blair: http://rebeccablairart.tumblr.com/
