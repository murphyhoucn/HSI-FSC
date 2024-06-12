
from EM_RN_model import Embedding
from EM_RN_model import RelationNetwork
from EM_RN_model import weights_init

from torchsummary import summary
net1 = Embedding().cuda()
# print(net1)
summary(net1, input_size=(1, 100, 28, 28))
summary(net1, input_size=(1, 100, 28, 28))

net2 = RelationNetwork().cuda()
summary(net2, input_size=(256, 5, 5))

