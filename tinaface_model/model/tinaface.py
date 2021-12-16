from keras import Model
from tinaface_model.model.resnet50 import ResNet50
from tinaface_model.model.fcn import FCN
from tinaface_model.model.inception import Inception
from tinaface_model.model.head import Head



class TinaFace(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet_block = ResNet50(return_intermediate=True)
        self.fcn_block = FCN()
        #
        self.inception_list = []
        self.head_list = []
        for i in range(4):
            head = Head()
            inception = Inception()
            self.inception_list.append(inception)
            self.head_list.append(head)

    def call(self, input_image):
        resnet_output = self.resnet_block(input_image)
        fcn_output = self.fcn_block(resnet_output)

        head_output_list = []
        for i in range(4):
            head_input = self.inception_list[i](fcn_output[i])
            head_output = self.head_list[i](head_input)
            head_output_list.append(head_output)
        return head_output_list



