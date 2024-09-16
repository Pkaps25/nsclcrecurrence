import monai

from monai.networks.nets import DenseNet


class NoduleRecurrenceClassifier(monai.networks.nets.densenet121):
    pass


small_net = DenseNet(spatial_dims=3, in_channels=1, out_channels=2, growth_rate=16)
