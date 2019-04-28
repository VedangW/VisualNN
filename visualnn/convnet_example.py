from convnet import ConvNet2D

layers_conv = [(200, 200, 12),
          (100, 100, 6),
          (50, 50, 12),
          (25, 25, 24)]

layers_dense = [100, 50, 25, 1]

convnet = ConvNet2D(layers_conv, layers_dense)
convnet.plot()
