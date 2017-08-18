try:
    from tensorflow.python import pywrap_tensorflow
    _tf_import_error = None
except ImportError as e:
    _tf_import_error = e

import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np


class ConvBnRelu(chainer.Chain):

    def __init__(self, depth, ksize, stride=1, pad=0, initialW=I.HeNormal()):
        super(ConvBnRelu, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, depth, ksize=ksize, stride=stride, pad=pad, initialW=initialW, nobias=True)
            self.bn = L.BatchNormalization(depth, decay=0.9997, eps=0.001, use_gamma=False)

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))

    def load_tf_checkpoint(self, reader, name):
        try:
            self.conv.W.data[...] = reader.get_tensor(name + '/weights').transpose(3, 2, 0, 1)
            self.bn.beta.data[...] = reader.get_tensor(name + '/BatchNorm/beta')
            self.bn.avg_mean[...] = reader.get_tensor(name + '/BatchNorm/moving_mean')
            self.bn.avg_var[...] = reader.get_tensor(name + '/BatchNorm/moving_variance')
        except Exception:
            print('failed at', name)
            raise


class TFLoadableChain(chainer.Chain):

    def load_tf_checkpoint(self, reader, path):
        for child in self.children():
            full_name = '{}/{}'.format(path, child.name)
            if isinstance(child, L.Convolution2D):
                try:
                    # Original model is 1001-way classification, but we only want 1000-way classification model
                    if full_name.startswith('InceptionV3/Logits/Conv2d_1c_1x1'):
                        start_index = 1
                    else:
                        start_index = 0
                    W = reader.get_tensor(full_name + '/weights').transpose(3, 2, 0, 1)
                    child.W.data[...] = W[start_index:]
                    if hasattr(child, 'b'):
                        b = reader.get_tensor(full_name + '/biases')
                        child.b.data[...] = b[start_index:]
                except Exception:
                    print('failed at', full_name)
                    raise
            else:
                child.load_tf_checkpoint(reader, full_name)


class InceptionBlock(TFLoadableChain):

    def __init__(self, depth, irregular_name=False):
        super(InceptionBlock, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_0a_1x1 = ConvBnRelu(64, 1)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                if irregular_name:
                    self.Branch_1.Conv2d_0b_1x1 = ConvBnRelu(48, 1)
                    self.Branch_1.Conv_1_0c_5x5 = ConvBnRelu(64, 5, pad=2)
                else:
                    self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(48, 1)
                    self.Branch_1.Conv2d_0b_5x5 = ConvBnRelu(64, 5, pad=2)

            self.Branch_2 = TFLoadableChain()
            with self.Branch_2.init_scope():
                self.Branch_2.Conv2d_0a_1x1 = ConvBnRelu(64, 1)
                self.Branch_2.Conv2d_0b_3x3 = ConvBnRelu(96, 3, pad=1)
                self.Branch_2.Conv2d_0c_3x3 = ConvBnRelu(96, 3, pad=1)

            self.Branch_3 = TFLoadableChain()
            with self.Branch_3.init_scope():
                self.Branch_3.Conv2d_0b_1x1 = ConvBnRelu(depth - (64 + 64 + 96), 1)

        self.irregular_name = irregular_name

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_0a_1x1(x)

        if self.irregular_name:
            br1 = self.Branch_1.Conv2d_0b_1x1(x)
            br1 = self.Branch_1.Conv_1_0c_5x5(br1)
        else:
            br1 = self.Branch_1.Conv2d_0a_1x1(x)
            br1 = self.Branch_1.Conv2d_0b_5x5(br1)

        br2 = self.Branch_2.Conv2d_0a_1x1(x)
        br2 = self.Branch_2.Conv2d_0b_3x3(br2)
        br2 = self.Branch_2.Conv2d_0c_3x3(br2)

        br3 = F.average_pooling_2d(x, 3, stride=1, pad=1)
        br3 = self.Branch_3.Conv2d_0b_1x1(br3)

        return F.concat((br0, br1, br2, br3))


class InceptionBlockVH(TFLoadableChain):

    def __init__(self, mid_depth):
        super(InceptionBlockVH, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_0a_1x1 = ConvBnRelu(192, 1)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(mid_depth, 1)
                self.Branch_1.Conv2d_0b_1x7 = ConvBnRelu(mid_depth, (1, 7), pad=(0, 3))
                self.Branch_1.Conv2d_0c_7x1 = ConvBnRelu(192, (7, 1), pad=(3, 0))

            self.Branch_2 = TFLoadableChain()
            with self.Branch_2.init_scope():
                self.Branch_2.Conv2d_0a_1x1 = ConvBnRelu(mid_depth, 1)
                self.Branch_2.Conv2d_0b_7x1 = ConvBnRelu(mid_depth, (7, 1), pad=(3, 0))
                self.Branch_2.Conv2d_0c_1x7 = ConvBnRelu(mid_depth, (1, 7), pad=(0, 3))
                self.Branch_2.Conv2d_0d_7x1 = ConvBnRelu(mid_depth, (7, 1), pad=(3, 0))
                self.Branch_2.Conv2d_0e_1x7 = ConvBnRelu(192, (1, 7), pad=(0, 3))

            self.Branch_3 = TFLoadableChain()
            with self.Branch_3.init_scope():
                self.Branch_3.Conv2d_0b_1x1 = ConvBnRelu(192, 1)

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_0a_1x1(x)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1 = self.Branch_1.Conv2d_0b_1x7(br1)
        br1 = self.Branch_1.Conv2d_0c_7x1(br1)

        br2 = self.Branch_2.Conv2d_0a_1x1(x)
        br2 = self.Branch_2.Conv2d_0b_7x1(br2)
        br2 = self.Branch_2.Conv2d_0c_1x7(br2)
        br2 = self.Branch_2.Conv2d_0d_7x1(br2)
        br2 = self.Branch_2.Conv2d_0e_1x7(br2)

        br3 = F.average_pooling_2d(x, 3, stride=1, pad=1)
        br3 = self.Branch_3.Conv2d_0b_1x1(br3)

        return F.concat((br0, br1, br2, br3))


class InceptionBlockExpanded(TFLoadableChain):

    def __init__(self, nametype):
        super(InceptionBlockExpanded, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_0a_1x1 = ConvBnRelu(320, 1)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(384, 1)
                self.Branch_1.Conv2d_0b_1x3 = ConvBnRelu(384, (1, 3), pad=(0, 1))
                if nametype == 'b':
                    self.Branch_1.Conv2d_0b_3x1 = ConvBnRelu(384, (3, 1), pad=(1, 0))
                else:
                    self.Branch_1.Conv2d_0c_3x1 = ConvBnRelu(384, (3, 1), pad=(1, 0))

            self.Branch_2 = TFLoadableChain()
            with self.Branch_2.init_scope():
                self.Branch_2.Conv2d_0a_1x1 = ConvBnRelu(448, 1)
                self.Branch_2.Conv2d_0b_3x3 = ConvBnRelu(384, 3, pad=1)
                self.Branch_2.Conv2d_0c_1x3 = ConvBnRelu(384, (1, 3), pad=(0, 1))
                self.Branch_2.Conv2d_0d_3x1 = ConvBnRelu(384, (3, 1), pad=(1, 0))

            self.Branch_3 = TFLoadableChain()
            with self.Branch_3.init_scope():
                self.Branch_3.Conv2d_0b_1x1 = ConvBnRelu(192, 1)

        self.nametype = nametype

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_0a_1x1(x)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1_a = self.Branch_1.Conv2d_0b_1x3(br1)
        if self.nametype == 'b':
            br1 = self.Branch_1.Conv2d_0b_3x1(br1)
        else:
            br1 = self.Branch_1.Conv2d_0c_3x1(br1)

        br2 = self.Branch_2.Conv2d_0a_1x1(x)
        br2 = self.Branch_2.Conv2d_0b_3x3(br2)
        br2_a = self.Branch_2.Conv2d_0c_1x3(br2)
        br2 = self.Branch_2.Conv2d_0d_3x1(br2)

        br3 = F.average_pooling_2d(x, 3, stride=1, pad=1)
        br3 = self.Branch_3.Conv2d_0b_1x1(br3)

        return F.concat((br0, br1_a, br1, br2_a, br2, br3))


class SubsamplingBlock1(TFLoadableChain):

    def __init__(self):
        super(SubsamplingBlock1, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_1a_1x1 = ConvBnRelu(384, 3, stride=2)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(64, 1)
                self.Branch_1.Conv2d_0b_3x3 = ConvBnRelu(96, 3, pad=1)
                self.Branch_1.Conv2d_1a_1x1 = ConvBnRelu(96, 3, stride=2)

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_1a_1x1(x)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1 = self.Branch_1.Conv2d_0b_3x3(br1)
        br1 = self.Branch_1.Conv2d_1a_1x1(br1)

        br2 = F.max_pooling_2d(x, 3, stride=2)

        return F.concat((br0, br1, br2))


class SubsamplingBlock2(TFLoadableChain):

    def __init__(self):
        super(SubsamplingBlock2, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_0a_1x1 = ConvBnRelu(192, 1)
                self.Branch_0.Conv2d_1a_3x3 = ConvBnRelu(320, 3, stride=2)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(192, 1)
                self.Branch_1.Conv2d_0b_1x7 = ConvBnRelu(192, (1, 7), pad=(0, 3))
                self.Branch_1.Conv2d_0c_7x1 = ConvBnRelu(192, (7, 1), pad=(3, 0))
                self.Branch_1.Conv2d_1a_3x3 = ConvBnRelu(192, 3, stride=2)

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_0a_1x1(x)
        br0 = self.Branch_0.Conv2d_1a_3x3(br0)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1 = self.Branch_1.Conv2d_0b_1x7(br1)
        br1 = self.Branch_1.Conv2d_0c_7x1(br1)
        br1 = self.Branch_1.Conv2d_1a_3x3(br1)

        br2 = F.max_pooling_2d(x, 3, stride=2)

        return F.concat((br0, br1, br2))


class AuxiliaryLogits(TFLoadableChain):

    def __init__(self, n_classes):
        super(AuxiliaryLogits, self).__init__()
        with self.init_scope():
            self.Conv2d_1b_1x1 = ConvBnRelu(128, 1)
            self.Conv2d_2a_5x5 = ConvBnRelu(768, 5, initialW=I.Normal(0.01))
            self.Conv2d_2b_1x1 = L.Convolution2D(None, n_classes, 1, initialW=I.Normal(0.001))

    def __call__(self, x):
        h = F.average_pooling_2d(x, 5, stride=3)
        h = self.Conv2d_1b_1x1(h)
        h = self.Conv2d_2a_5x5(h)
        h = self.Conv2d_2b_1x1(h)
        return h[..., 0, 0]


class InceptionV3(TFLoadableChain):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.8, enable_aux=False):
        super(InceptionV3, self).__init__()
        with self.init_scope():
            self.Conv2d_1a_3x3 = ConvBnRelu(32, 3, stride=2)
            self.Conv2d_2a_3x3 = ConvBnRelu(32, 3)
            self.Conv2d_2b_3x3 = ConvBnRelu(64, 3, pad=1)
            self.Conv2d_3b_1x1 = ConvBnRelu(80, 1)
            self.Conv2d_4a_3x3 = ConvBnRelu(192, 3)

            self.Mixed_5b = InceptionBlock(256)
            self.Mixed_5c = InceptionBlock(288, irregular_name=True)
            self.Mixed_5d = InceptionBlock(288)
            self.Mixed_6a = SubsamplingBlock1()
            self.Mixed_6b = InceptionBlockVH(128)
            self.Mixed_6c = InceptionBlockVH(160)
            self.Mixed_6d = InceptionBlockVH(160)
            self.Mixed_6e = InceptionBlockVH(192)
            self.Mixed_7a = SubsamplingBlock2()
            self.Mixed_7b = InceptionBlockExpanded('b')
            self.Mixed_7c = InceptionBlockExpanded('c')

            self.Logits = TFLoadableChain()
            with self.Logits.init_scope():
                self.Logits.Conv2d_1c_1x1 = L.Convolution2D(None, num_classes, 1)

            if enable_aux:
                self.AuxLogits = AuxiliaryLogits(num_classes)

        self.dropout_rate = 1 - dropout_keep_prob
        self.enable_aux = enable_aux
        self.cnt = 0

    def __call__(self, x):
        # Preprocessing
        with chainer.cuda.get_device_from_array(x, x.data):
            h = (x / 255 - .5) * 2
        h = self.Conv2d_1a_3x3(h)
        h = self.Conv2d_2a_3x3(h)
        h = self.Conv2d_2b_3x3(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.Conv2d_3b_1x1(h)
        h = self.Conv2d_4a_3x3(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.Mixed_5b(h)
        h = self.Mixed_5c(h)
        h = self.Mixed_5d(h)

        h = self.Mixed_6a(h)
        h = self.Mixed_6b(h)
        h = self.Mixed_6c(h)
        h = self.Mixed_6d(h)
        h = self.Mixed_6e(h)
        if self.enable_aux:
            aux_logits = self.AuxLogits(h)

        h = self.Mixed_7a(h)
        h = self.Mixed_7b(h)
        h = self.Mixed_7c(h)

        h = F.average_pooling_2d(h, 8)
        h = F.dropout(h, self.dropout_rate)
        h = self.Logits.Conv2d_1c_1x1(h)
        h = h[..., 0, 0]

        if self.enable_aux:
            return h, aux_logits
        return h


def load_inception_v3(checkpoint_path, enable_aux=False):
    if _tf_import_error is not None:
        raise RuntimeError('could not import tensorflow; the import error as follows:\n' + str(_tf_import_error))
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    model = InceptionV3(enable_aux=enable_aux)
    with chainer.no_backprop_mode():
        model(np.random.randn(2, 3, 299, 299).astype('f'))  # initialize params
    model.load_tf_checkpoint(reader, 'InceptionV3')
    return model
