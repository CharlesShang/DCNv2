## Deformable Convolutional Networks V2 with Pytorch
```bash
    .\make  # build
    python test.py # run gradient check 
```
### Known issues:

-[ ] Gradient check w.r.t offset. 
-[ ] Backward is not reentrant.

This is adaption of the official [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op).
I have ran the gradient check for many times with DOUBLE type. Every tensor **except offset** passes. 
However, when I set the offset to 0.5, it passes. I'm still wondering what cause this problem. Is it because some
non-differential points? 

Another issue is that it raises `RuntimeError: Backward is not reentrant`. However, the error is very small `(<1e-7)`, 
so it may not be a serious problem (?)

Please post an issue or PR if you have any comments.
    