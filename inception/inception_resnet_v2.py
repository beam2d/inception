try:
    from tensorflow.python import pywrap_tensorflow
    _tf_import_error = None
except ImportError as e:
    _tf_import_error = e

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np

from .conv import ConvBnRelu
from .tf_loadable_chain import TFLoadableChain, TFLoadableRepeat


# This function is used for runtime initialization of the Convolution2D layer at the end of each repeated block. The
# lazy initialization is used to keep the initialization code simpler and require less manual unit size specification.
def lazy_init_conv_to_join(block, x):
    if not hasattr(block, 'Conv2d_1x1'):
        with block.init_scope():
            block.Conv2d_1x1 = L.Convolution2D(x.shape[1], 1, initialW=I.HeNormal())
        if isinstance(x.data, cuda.ndarray):
            block.Conv2d_1x1.to_gpu(x.data.device)


class Block35(TFLoadableChain):

    """35x35 resnet block."""

    def __init__(self, scale=1.0, activation_fn=F.relu):
        super(Block35, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_1x1 = ConvBnRelu(32, 1)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(32, 1)
                self.Branch_1.Conv2d_0b_3x3 = ConvBnRelu(32, 3, pad=1)

            self.Branch_2 = TFLoadableChain()
            with self.Branch_2.init_scope():
                self.Branch_2.Conv2d_0a_1x1 = ConvBnRelu(32, 1)
                self.Branch_2.Conv2d_0b_3x3 = ConvBnRelu(48, 3, pad=1)
                self.Branch_2.Conv2d_0c_3x3 = ConvBnRelu(64, 3, pad=1)

            # NOTE: Conv2d_1x1 is built at the first iteration

        self.scale = scale
        self.activation_fn = activation_fn

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_1x1(x)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1 = self.Branch_1.Conv2d_0b_3x3(br1)

        br2 = self.Branch_2.Conv2d_0a_1x1(x)
        br2 = self.Branch_2.Conv2d_0b_3x3(br2)
        br2 = self.Branch_2.Conv2d_0c_3x3(br2)

        mixed = F.concat((br0, br1, br2), axis=1)

        lazy_init_conv_to_join(self, x)
        up = self.Conv2d_1x1(mixed)

        x += self.scale * up
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class Block17(TFLoadableChain):

    def __init__(self, scale=1.0, activation_fn=F.relu):
        super(Block17, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_1x1 = ConvBnRelu(192, 1)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(128, 1)
                self.Branch_1.Conv2d_0b_1x7 = ConvBnRelu(160, (1, 7), pad=(0, 3))
                self.Branch_1.Conv2d_0c_7x1 = ConvBnRelu(192, (7, 1), pad=(3, 0))

            # NOTE: Conv2d_1x1 is built at the first iteration
                
        self.scale = scale
        self.activation_fn = activation_fn

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_1x1(x)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1 = self.Branch_1.Conv2d_0b_1x7(br1)
        br1 = self.Branch_1.Conv2d_0c_7x1(br1)

        mixed = F.concat((br0, br1), axis=1)

        lazy_init_conv_to_join(self, x)
        up = self.Conv2d_1x1(mixed)

        x += self.scale * up
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class Block8(TFLoadableChain):

    def __init__(self, scale=1.0, activation_fn=F.relu):
        super(Block8, self).__init__()
        with self.init_scope():
            self.Branch_0 = TFLoadableChain()
            with self.Branch_0.init_scope():
                self.Branch_0.Conv2d_1x1 = ConvBnRelu(192, 1)

            self.Branch_1 = TFLoadableChain()
            with self.Branch_1.init_scope():
                self.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(192, 1)
                self.Branch_1.Conv2d_0b_1x3 = ConvBnRelu(224, (1, 3), pad=(0, 1))
                self.Branch_1.Conv2d_0c_3x1 = ConvBnRelu(256, (3, 1), pad=(1, 0))

            # NOTE: Conv2d_1x1 is built at the first iteration
                
        self.scale = scale
        self.activation_fn = activation_fn

    def __call__(self, x):
        br0 = self.Branch_0.Conv2d_1x1(x)

        br1 = self.Branch_1.Conv2d_0a_1x1(x)
        br1 = self.Branch_1.Conv2d_0b_1x3(br1)
        br1 = self.Branch_1.Conv2d_0c_3x1(br1)

        mixed = F.concat((br0, br1), axis=1)

        lazy_init_conv_to_join(self, x)
        up = self.Conv2d_1x1(mixed)

        x += self.scale * up
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionResnetV2(TFLoadableChain):

    def __init__(self, dropout_keep_prob=0.8, enable_aux=False):
        super(InceptionResnetV2, self).__init__()
        with self.init_scope():
            self.Conv2d_1a_3x3 = ConvBnRelu(32, 3, stride=2)
            self.Conv2d_2a_3x3 = ConvBnRelu(32, 3)
            self.Conv2d_2b_3x3 = ConvBnRelu(64, 3, pad=1)
            self.Conv2d_3b_1x1 = ConvBnRelu(80, 1)
            self.Conv2d_4a_3x3 = ConvBnRelu(192, 3)

            self.Mixed_5b = TFLoadableChain()
            with self.Mixed_5b.init_scope():
                self.Mixed_5b.Branch_0 = TFLoadableChain()
                with self.Mixed_5b.Branch_0.init_scope():
                    self.Mixed_5b.Branch_0.Conv2d_1x1 = ConvBnRelu(96, 1)

                self.Mixed_5b.Branch_1 = TFLoadableChain()
                with self.Mixed_5b.Branch_1.init_scope():
                    self.Mixed_5b.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(48, 1)
                    self.Mixed_5b.Branch_1.Conv2d_0b_5x5 = ConvBnRelu(64, 5, pad=2)

                self.Mixed_5b.Branch_2 = TFLoadableChain()
                with self.Mixed_5b.Branch_2.init_scope():
                    self.Mixed_5b.Branch_2.Conv2d_0a_1x1 = ConvBnRelu(64, 1)
                    self.Mixed_5b.Branch_2.Conv2d_0b_3x3 = ConvBnRelu(96, 3, pad=1)
                    self.Mixed_5b.Branch_2.Conv2d_0c_3x3 = ConvBnRelu(96, 3, pad=1)

                self.Mixed_5b.Branch_3 = TFLoadableChain()
                with self.Mixed_5b.Branch_3.init_scope():
                    self.Mixed_5b.Branch_3.Conv2d_0b_1x1 = ConvBnRelu(64, 1)

            self.Repeat = TFLoadableRepeat(lambda: Block35(scale=0.17), 10, 'block35')

            self.Mixed_6a = TFLoadableChain()
            with self.Mixed_6a.init_scope():
                self.Mixed_6a.Branch_0 = TFLoadableChain()
                with self.Mixed_6a.Branch_0.init_scope():
                    self.Mixed_6a.Branch_0.Conv2d_1a_3x3 = ConvBnRelu(384, 3, stride=2)

                self.Mixed_6a.Branch_1 = TFLoadableChain()
                with self.Mixed_6a.Branch_1.init_scope():
                    self.Mixed_6a.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(256, 1)
                    self.Mixed_6a.Branch_1.Conv2d_0b_3x3 = ConvBnRelu(256, 3, pad=1)
                    self.Mixed_6a.Branch_1.Conv2d_1a_3x3 = ConvBnRelu(384, 3, stride=2)

            self.Repeat_1 = TFLoadableRepeat(lambda: Block17(scale=0.10), 20, 'block17')

            self.Mixed_7a = TFLoadableChain()
            with self.Mixed_7a.init_scope():
                self.Mixed_7a.Branch_0 = TFLoadableChain()
                with self.Mixed_7a.Branch_0.init_scope():
                    self.Mixed_7a.Branch_0.Conv2d_0a_1x1 = ConvBnRelu(256, 1)
                    self.Mixed_7a.Branch_0.Conv2d_1a_3x3 = ConvBnRelu(384, 3, stride=2)

                self.Mixed_7a.Branch_1 = TFLoadableChain()
                with self.Mixed_7a.Branch_1.init_scope():
                    self.Mixed_7a.Branch_1.Conv2d_0a_1x1 = ConvBnRelu(256, 1)
                    self.Mixed_7a.Branch_1.Conv2d_1a_3x3 = ConvBnRelu(288, 3, stride=2)

                self.Mixed_7a.Branch_2 = TFLoadableChain()
                with self.Mixed_7a.Branch_2.init_scope():
                    self.Mixed_7a.Branch_2.Conv2d_0a_1x1 = ConvBnRelu(256, 1)
                    self.Mixed_7a.Branch_2.Conv2d_0b_3x3 = ConvBnRelu(288, 3, pad=1)
                    self.Mixed_7a.Branch_2.Conv2d_1a_3x3 = ConvBnRelu(320, 3, stride=2)

            self.Repeat_2 = TFLoadableRepeat(lambda: Block8(scale=0.20), 9, 'block8')
            self.Block8 = Block8(activation_fn=None)

            self.Conv2d_7b_1x1 = ConvBnRelu(1536, 1)

            if enable_aux:
                self.AuxLogits = TFLoadableChain()
                with self.AuxLogits.init_scope():
                    self.AuxLogits.Conv2d_1b_1x1 = ConvBnRelu(128, 1)
                    self.AuxLogits.Conv2d_2a_5x5 = ConvBnRelu(768, 5)
                    self.AuxLogits.Logits = L.Linear(1000)

            self.Logits = TFLoadableChain(final_layer_name='Logits')
            with self.Logits.init_scope():
                self.Logits.Logits = L.Linear(1000)

        self.dropout_rate = 1 - dropout_keep_prob
        self.enable_aux = enable_aux

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

        h0 = self.Mixed_5b.Branch_0.Conv2d_1x1(h)
        h1 = self.Mixed_5b.Branch_1.Conv2d_0a_1x1(h)
        h1 = self.Mixed_5b.Branch_1.Conv2d_0b_5x5(h1)
        h2 = self.Mixed_5b.Branch_2.Conv2d_0a_1x1(h)
        h2 = self.Mixed_5b.Branch_2.Conv2d_0b_3x3(h2)
        h2 = self.Mixed_5b.Branch_2.Conv2d_0c_3x3(h2)
        h3 = F.average_pooling_2d(h, 3, stride=1, pad=1)
        h3 = self.Mixed_5b.Branch_3.Conv2d_0b_1x1(h3)
        h = F.concat((h0, h1, h2, h3), axis=1)
        del h0, h1, h2, h3

        h = self.Repeat(h)

        h0 = self.Mixed_6a.Branch_0.Conv2d_1a_3x3(h)
        h1 = self.Mixed_6a.Branch_1.Conv2d_0a_1x1(h)
        h1 = self.Mixed_6a.Branch_1.Conv2d_0b_3x3(h1)
        h1 = self.Mixed_6a.Branch_1.Conv2d_1a_3x3(h1)
        h2 = F.max_pooling_2d(h, 3, stride=2)
        h = F.concat((h0, h1, h2), axis=1)
        del h0, h1, h2

        h = self.Repeat_1(h)

        if self.enable_aux:
            aux = F.average_pooling_2d(h, h.shape[2:], stride=3)
            aux = self.AuxLogits.Conv2d_1a_3x3(aux)
            aux = self.AuxLogits.Conv2d_1b_1x1(aux)
            aux = self.AuxLogits.Conv2d_2a_5x5(aux)
            aux = self.AuxLogits.Logits(aux)

        h0 = self.Mixed_7a.Branch_0.Conv2d_0a_1x1(h)
        h0 = self.Mixed_7a.Branch_0.Conv2d_1a_3x3(h0)
        h1 = self.Mixed_7a.Branch_1.Conv2d_0a_1x1(h)
        h1 = self.Mixed_7a.Branch_1.Conv2d_1a_3x3(h1)
        h2 = self.Mixed_7a.Branch_2.Conv2d_0a_1x1(h)
        h2 = self.Mixed_7a.Branch_2.Conv2d_0b_3x3(h2)
        h2 = self.Mixed_7a.Branch_2.Conv2d_1a_3x3(h2)
        h3 = F.max_pooling_2d(h, 3, stride=2)
        h = F.concat((h0, h1, h2, h3), axis=1)
        del h0, h1, h2, h3

        h = self.Repeat_2(h)
        h = self.Block8(h)
        h = self.Conv2d_7b_1x1(h)

        h = F.average_pooling_2d(h, h.shape[2:])
        h = F.dropout(h, self.dropout_rate)
        h = self.Logits.Logits(h)

        if self.enable_aux:
            return h, aux
        return h


def load_inception_resnet_v2(checkpoint_path, enable_aux=False):
    if _tf_import_error is not None:
        raise RuntimeError('could not import tensorflow; the import error as follows:\n' + str(_tf_import_error))
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    model = InceptionResnetV2(enable_aux=enable_aux)
    with chainer.no_backprop_mode():
        model(np.random.randn(2, 3, 299, 299).astype('f'))  # initialize params
    model.load_tf_checkpoint(reader, 'InceptionResnetV2')
    return model
