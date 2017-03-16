# Import keras packages
from keras.applications import inception_v3, vgg16
from keras.models import Model

def load_distributed_model(type="inception", weights="imagenet", bottom="full", include_top=True):
    """Load distributed CNN model such as GoogLeNet-v3 and VGG16 with arbitary depth of the model. 

    :param type: Any model types of 'inception'(GoogLeNet-v3) and 'vgg'(VGG16). 
    :param weights: Any model weights of 'imagenet'(pre-trained) and None(non-pre-trained). 
    :param bottom: Name of the bottom layer which a returned model will have. 

        - If you don't know layer name, check distributed model summary on Keras platform by using summary() method. 
        - bettom == 'full' means any layer will be cut off from the model. 

    :param include_top: Flag to include the regression section in the model like 'global average pooling' and 'dense' layers. 
    :return: Keras model. 
    """

    base = None
    if type == "inception":
        base = inception_v3.InceptionV3(weights=weights, include_top=include_top)
    elif type == "vgg":
        base = vgg16.VGG16(weights=weights, include_top=include_top)
    else:
        print "[E]Not available type '%s'. " % type
        quit()

    output_layer = base.output if bottom == "full" else base.get_layer(name=bottom)

    return Model(input=base.input, output=output_layer)

if __name__ == '__main__':
    main()