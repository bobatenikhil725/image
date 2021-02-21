# image harmonization 
In this project, we are training a convolutional neural network
in an adversarial way for image harmonization. Given a composite
image and a foreground mask as the input, our model directly out-
puts a harmonized image, where the contents are the same as the
input but with adjusted appearances on the foreground region. The
foreground appearances can be adjusted accordingly to generate a
realistic composite image. Toward this end, we train two neural
networks in an adversarial way to capture the context of the input
image and to reconstruct the harmonized image using the learned
representations.

# Dataset
For the image harmonization task, we have collected images from
the MSCOCO dataset which have ground-truth foreground masks,
and then applied color transfer between random foreground pairs
with the same semantic labels. While the image after a foreground
adjustment is used as the input, the original image is used as the
ground-truth

# Methodology 
The generator is an encoder-decoder model using a U-
Net architecture. The model takes a source image and generates a
target image. It does this by rst downsampling or encoding the in-
put image down to a bottleneck layer, then upsampling or decoding
the bottleneck representation to the size of the output image. The
U-Net architecture means that skip-connections are added between
the encoding layers and the corresponding decoding layers, forming
a U-shape./

The discriminator design is based on the eective recep-
tive eld of the model, which denes the relationship between one
output of the model to the number of pixels in the input image. This
is called a PatchGAN model and is carefully designed so that each
output prediction of the model maps to a 70×70 square or patch
of the input image.




