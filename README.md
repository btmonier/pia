# piar

`piar` turns high resolution images into low resolution, small palette outputs in
the style of pixel art. It implements the method described in:

> Gerstner, T., DeCarlo, D., Alexa, M., Finkelstein, A., Gingold, Y., & Nealen, A.
> (2012). *Pixelated Image Abstraction*. Proceedings of the International
> Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012). [`10.1016/j.cag.2012.12.007`](https://doi.org/10.1016/j.cag.2012.12.007)

Rather than downsampling and then quantizing colors, the algorithm iteratively
solves for both at once: a modified SLIC segmentation maps regions of the input
to output pixels (one superpixel per output pixel) while mass constrained
deterministic annealing grows an optimal palette in CIELAB space. The result
preserves salient features and colors far better than naive resizing.

