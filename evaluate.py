import chainer
import fire

from inception import inception_v3


_MODEL_LOADERS = {
    'InceptionV3': inception_v3.load_inception_v3
}


def evaluate_inception(model_type, checkpoint_path, dataset_path, dataset_root='/', gpu=0):
    loader = _MODEL_LOADERS[model_type]
    model = chainer.links.Classifier(loader(checkpoint_path))
    if gpu >= 0:
        model.to_gpu(gpu)

    dataset = chainer.datasets.LabeledImageDataset(dataset_path, root=dataset_root)
    iterator = chainer.iterators.SerialIterator(dataset, 100, repeat=False, shuffle=False)
    evaluator = chainer.training.extensions.Evaluator(iterator, model, device=gpu)
    result = evaluator()

    print(result)


if __name__ == '__main__':
    fire.Fire(evaluate_inception)
