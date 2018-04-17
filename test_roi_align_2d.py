# -*- coding: utf-8 -*-
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

from roi_align_2d import ROIAlign2D, roi_align_2d


class TestROIAlign2D(unittest.TestCase):
    def setUp(self):
        N = 3
        n_channels = 256
        self.x = numpy.arange(
            N * n_channels * 12 * 8, dtype=numpy.float32).reshape(
                (N, n_channels, 12, 8))
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        self.x = self.x.astype(numpy.float32)
        self.rois = numpy.array(
            [[0, 1, 1, 6, 6], [2, 6, 2, 7, 11], [1, 3, 1, 5, 10],
             [0, 3, 3, 3, 3]],
            dtype=numpy.float32)
        self.rois = numpy.tile(self.rois, (15, 1))
        n_rois = self.rois.shape[0]
        self.outh, self.outw = 5, 7
        self.spatial_scale = 0.6
        self.gy = numpy.random.uniform(
            -1, 1, (n_rois, n_channels, self.outh, self.outw)).astype(
                numpy.float32)
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, roi_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = roi_align_2d(
            x,
            rois,
            outh=self.outh,
            outw=self.outw,
            spatial_scale=self.spatial_scale)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois))

    @attr.gpu
    def test_forward_cpu_gpu_equal(self):
        # cpu
        x_cpu = chainer.Variable(self.x)
        rois_cpu = chainer.Variable(self.rois)
        y_cpu = roi_align_2d(
            x_cpu,
            rois_cpu,
            outh=self.outh,
            outw=self.outw,
            spatial_scale=self.spatial_scale)

        # gpu
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        rois_gpu = chainer.Variable(cuda.to_gpu(self.rois))
        y_gpu = roi_align_2d(
            x_gpu,
            rois_gpu,
            outh=self.outh,
            outw=self.outw,
            spatial_scale=self.spatial_scale)
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, roi_data, y_grad):
        gradient_check.check_backward(
            ROIAlign2D(
                outh=self.outh,
                outw=self.outw,
                spatial_scale=self.spatial_scale), (x_data, roi_data),
            y_grad,
            no_grads=[False, True],
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois), cuda.to_gpu(self.gy))

    def test_caffe2_equal(self):
        try:
            from caffe2.python import core, workspace
        except:
            assert False

        op = core.CreateOperator(
            "RoIAlign", ["x", "rois"], ["y"],
            order="NCHW",
            pooled_h=self.outh,
            pooled_w=self.outw,
            spatial_scale=self.spatial_scale,
            sampling_ratio=1)
        workspace.ResetWorkspace()
        workspace.FeedBlob("x", self.x)
        workspace.FeedBlob("rois", self.rois)
        workspace.RunOperatorOnce(op)
        y_caffe = workspace.FetchBlob("y")

        x = chainer.Variable(self.x)
        rois = chainer.Variable(self.rois)
        y = roi_align_2d(
            x,
            rois,
            outh=self.outh,
            outw=self.outw,
            spatial_scale=self.spatial_scale)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)
        gradient_check.assert_allclose(y_caffe, y_data)

    def test_forward_cpu_1_2_equal(self):
        # cpu
        x = chainer.Variable(self.x)
        rois = chainer.Variable(self.rois)
        import time
        start = time.time()
        y1 = roi_align_2d(
            x,
            rois,
            outh=self.outh,
            outw=self.outw,
            spatial_scale=self.spatial_scale)
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        start = time.time()
        y2, = ROIAlign2D(self.outh, self.outw,
                         self.spatial_scale).forward_cpu2((self.x, self.rois))
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        testing.assert_allclose(y1.data, y2.data)


testing.run_module(__name__, __file__)
