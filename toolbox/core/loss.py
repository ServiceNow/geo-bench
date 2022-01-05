"""Module for providing evaluation losses. 

"""


class Loss(object):
    def __call__(self, label, prediction):
        raise NotImplemented()

    @property
    def name(self):
        return self.__class__.__name__.lower()


class Accuracy(Loss):
    def __call__(self, prediction, label):
        return float(label != prediction)


class SegmentationAccuracy(Loss):
    pass
