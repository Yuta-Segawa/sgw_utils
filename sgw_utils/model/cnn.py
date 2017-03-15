# Import keras packages
from keras.applications import inception_v3, vgg16
from keras.models import Model

def load_distributed_model(type="inception", weights="imagenet", bottom="full", include_top=True):

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