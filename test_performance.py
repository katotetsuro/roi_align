import timeit
import chainer.functions as F
from roi_align_2d import roi_align_2d
import chainer


def pooling_forward(x, roi, outh, outw, spatial_scale):
    y = F.roi_pooling_2d(x, roi, outh, outw, spatial_scale)
    return y

def align_forward(x, roi, outh, outw, spatial_scale):
    y = roi_align_2d(x, roi, outh, outw, spatial_scale)
    return y


if __name__ == '__main__':
    xp = chainer.cuda.cupy
    x = xp.zeros((1, 512, 224, 224), dtype=xp.float32)
    roi = xp.array((0, 0, 0, 200, 200), dtype=xp.float32).reshape(1, 5)
    roi = xp.concatenate([roi, roi, roi])
    outh = 14
    outw = 14
    spatial_scale = 1.
    n = 1000
    timeit.timeit(lambda :pooling_forward(x, roi, outh, outw, spatial_scale), number=1)
    print(timeit.timeit(lambda :pooling_forward(x, roi, outh, outw, spatial_scale), number=n))
    # compile
    timeit.timeit(lambda :align_forward(x, roi, outh, outw, spatial_scale), number=1)
    print(timeit.timeit(lambda :align_forward(x, roi, outh, outw, spatial_scale), number=n))

    y = pooling_forward(x, roi, outh, outw, spatial_scale)
    print(timeit.timeit(lambda :y.backward(), number=n))
    y = align_forward(x, roi, outh, outw, spatial_scale)
    print(timeit.timeit(lambda :y.backward(), number=n))
