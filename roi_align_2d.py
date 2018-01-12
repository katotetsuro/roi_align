# Mask R-CNN
# Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
# https://arxiv.org/abs/1703.06870

import numpy
import six

from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(numpy.floor(size * stride))
    end = int(numpy.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start


class ROIAlign2D(function_node.FunctionNode):

    """RoI align over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        # `numpy.zeros` needs to be used because the arrays can be
        # returned without having some of its values updated.
        top_data = numpy.zeros((n_rois, channels, self.outh, self.outw),
                               dtype=numpy.float32)
        # store center (x, y) of each bins, floating point
        self.center_data = numpy.zeros((n_rois, 2, self.outh, self.outw), numpy.float32)

        for i_roi in six.moves.range(n_rois):
            idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            xmin = xmin * self.spatial_scale
            xmax = xmax * self.spatial_scale
            ymin = ymin * self.spatial_scale
            ymax = ymax * self.spatial_scale
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            #center_y = np.indices((self.outh, self.outw))
            #center_y = (center_y + 0.5) * strideh + ymin
            #center_x = (center_x + 0.5) * stridew + xmin
            #centers = np.array((center_y, center_x))
            #self.center_data[i_roi, :] = centers

            #x00 = np.floor(centers).astype(np.int32)
            #p, q = centers - x00
            #x00[i_roi, 0, :, :
            
            
            for outh in six.moves.range(self.outh):
                for outw in six.moves.range(self.outw):

                    cy = (outh + 0.5) * strideh + ymin
                    cx = (outw + 0.5) * stridew + xmin

                    # adjacents_, (0,0),(0,1),(1,0),(1,1)的なもの
                    # weights (1-p)*(1-q), (1-p)*q, p*(1-q), p*q 的なもの
                    # だけどこの内積はけっこうちゃんと考えないとやばいな
                    x00 = numpy.array((cy, cx), dtype=numpy.int32)
                    p, q = numpy.array((cy, cx)) - x00
                    bound = (height-1, width-1)
                    x10 = numpy.maximum(x00 + (1,0), bound)
                    x01 = numpy.maximum(x00 + (0, 1), bound)
                    x11 = numpy.maximum(x00 + (1, 1), bound) 

                    roi_data = bottom_data[int(idx), :, x00[0], x00[1]] * (1-p)*(1-q) \
                                + bottom_data[int(idx), :, x10[0], x10[1]] * p * (1-q) \
                                + bottom_data[int(idx), :, x01[0], x01[1]] * (1-p) * q \
                                + bottom_data[int(idx), :, x11[0], x11[1]] * p * q 
                    top_data[i_roi, :, outh, outw] = roi_data
                    self.center_data[i_roi, :, outh, outw] = [cx, cy]
        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=numpy.float32)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data, int32 argmax_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_rois[num * 5 + 0];
            int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);
            int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);
            int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);
            int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);
            float bin_size_h = static_cast<float>(roi_height)
                           / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                           / static_cast<float>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                          * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                          * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                        * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                        * bin_size_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = (roi_batch_ind * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * width + w;
                    if (bottom_data[data_offset + bottom_index] > maxval) {
                        maxval = bottom_data[data_offset + bottom_index];
                        maxidx = bottom_index;
                    }
                }
            }
            top_data = maxval;
            argmax_data = maxidx;
            ''', 'roi_pooling_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_rois, top_data,
          self.argmax_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        n_rois = bottom_rois.shape[0]
        bottom_delta = numpy.zeros(self._bottom_data_shape, numpy.float32)

        for i_roi in six.moves.range(n_rois):
            idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            idx = int(idx)
            xmin = xmin * self.spatial_scale
            xmax = xmax * self.spatial_scale
            ymin = ymin * self.spatial_scale
            ymax = ymax * self.spatial_scale
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)

            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)

            #gyの各成分に対して、対応する4近傍点にgradを加算する
            c, rows, cols = gy[0].shape
            assert gy[0].shape == self.center_data.shape
            for y in rows:
                for x in cols:
                    cy, cx = self.center_data[i_roi, :, y, x]
                    x00 = numpy.array((cy, cx), dtype=numpy.int32)
                    p, q = numpy.array((cy, cx)) - x00
                    bound = (height-1, width-1)
                    x10 = numpy.maximum(x00 + (1,0), bound)
                    x01 = numpy.maximum(x00 + (0, 1), bound)
                    x11 = numpy.maximum(x00 + (1, 1), bound) 
                    bottom_delta[idx, :, x00[0], x00[1]] += (1-p)*(1-q) * gy[0][i_roi, :, y, x]
                    bottom_delta[idx, :, x10[0], x10[1]] += p*(1-q) * gy[0][i_roi, :, y, x]
                    bottom_delta[idx, :, x01[0], x01[1]] += (1-p)*q * gy[0][i_roi, :, y, x]
                    bottom_delta[idx, :, x11[0], x11[1]] += p*q * gy[0][i_roi, :, y, x]

        return bottom_delta, None

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 argmax_data, int32 num_rois,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, raw float32 bottom_rois
            ''',
            'float32 bottom_diff',
            '''
            int w = i % width;
            int h = (i / width) % height;
            int c = (i / (width * height)) % channels;
            int num = i / (width * height * channels);

            float gradient = 0;
            // Accumulate gradient over all ROIs that pooled this element
            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
                // Skip if ROI's batch index doesn't match num
                if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
                    continue;
                }

                int roi_start_w = round(bottom_rois[roi_n * 5 + 1]
                                        * spatial_scale);
                int roi_start_h = round(bottom_rois[roi_n * 5 + 2]
                                        * spatial_scale);
                int roi_end_w = round(bottom_rois[roi_n * 5 + 3]
                                      * spatial_scale);
                int roi_end_h = round(bottom_rois[roi_n * 5 + 4]
                                      * spatial_scale);

                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                     h >= roi_start_h && h <= roi_end_h);
                if (!in_roi) {
                    continue;
                }

                int offset = (roi_n * channels + c) * pooled_height
                             * pooled_width;

                // Compute feasible set of pooled units that could have pooled
                // this bottom unit

                // Force malformed ROIs to be 1x1
                int roi_width = max(roi_end_w - roi_start_w + 1, 1);
                int roi_height = max(roi_end_h - roi_start_h + 1, 1);

                float bin_size_h = static_cast<float>(roi_height)
                               / static_cast<float>(pooled_height);
                float bin_size_w = static_cast<float>(roi_width)
                               / static_cast<float>(pooled_width);

                int phstart = floor(static_cast<float>(h - roi_start_h)
                                    / bin_size_h);
                int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                                 / bin_size_h);
                int pwstart = floor(static_cast<float>(w - roi_start_w)
                                    / bin_size_w);
                int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                                 / bin_size_w);

                phstart = min(max(phstart, 0), pooled_height);
                phend = min(max(phend, 0), pooled_height);
                pwstart = min(max(pwstart, 0), pooled_width);
                pwend = min(max(pwend, 0), pooled_width);

                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int index_ = ph * pooled_width + pw + offset;
                        if (argmax_data[index_] == (h * width + w)) {
                            gradient += top_diff[index_];
                        }
                    }
                }
            }
            bottom_diff = gradient;
            ''', 'roi_pooling_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois.shape[0], self.spatial_scale,
          channels, height, width, self.outh, self.outw,
          bottom_rois, bottom_diff)

        return bottom_diff, None


def roi_align_2d(x, rois, outh, outw, spatial_scale):
    """Spatial Region of Interest (ROI) align function.

    This function acts similarly to :class:`~functions.RoIPooling2D`, but
    #TODO explain

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 5), and each datum is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        outh (int): Height of output image after pooled.
        outw (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return ROIAlign2D(outh, outw, spatial_scale).apply((x, rois))[0]
