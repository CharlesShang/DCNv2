## Deformable Convolutional Networks V2 with Pytorch

### Build
```bash
    ./make.sh         # build
    python test.py    # run examples and gradient check 
```

### An Example
```python
    from dcn_v2 import DCN
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
    output = dcn(input)
    print(output.shape)
```

### Known Issues:

- [x] Gradient check w.r.t offset (solved)
- [ ] Backward is not reentrant (minor)

This is an adaption of the official [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op).

<s>I have ran the gradient check for many times with DOUBLE type. Every tensor **except offset** passes.
However, when I set the offset to 0.5, it passes. I'm still wondering what cause this problem. Is it because some
non-differential points? </s>

Update: all gradient check passes with double precision. 

Another issue is that it raises `RuntimeError: Backward is not reentrant`. However, the error is very small (`<1e-7` for 
float `<1e-15` for double), 
so it may not be a serious problem (?)

Please post an issue or PR if you have any comments.
    