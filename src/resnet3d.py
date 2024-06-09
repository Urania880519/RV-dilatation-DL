import tensorflow as tf
import einops
#dimensions


class Conv3d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq= tf.keras.Sequential([
            #spatial decomposition
            tf.keras.layers.Conv3D(
                filters= filters, 
                kernel_size= (1, kernel_size[1], kernel_size[2]),
                padding= padding
            ),
            #temporal decomposition
            tf.keras.layers.Conv3D(
                filters= filters, 
                kernel_size= (kernel_size[0],1,1),
                padding= padding
            )
        ])
    def call(self, x):
        return self.seq(x)

class MainRes(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, projection= None):
        super().__init__()
        self.projection= projection
        self.seq= tf.keras.Sequential([
            Conv3d(
                filters= filters,
                kernel_size=kernel_size,
                padding="same"
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            Conv3d(
                filters=filters,
                kernel_size= kernel_size,
                padding= "same"
            )
        ])
    def call(self,x):
        out= self.seq(x)
        res= x
        if out.shape[-1] != x.shape[-1]:
            self.projection= Projection
            res= self.projection(out.shape[-1])(res)
        out= out+ res
        return out


class Projection(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.seq= tf.keras.Sequential([
            tf.keras.layers.Dense(units),
            tf.keras.layers.LayerNormalization()
        ])
    def call(self, x):
        return self.seq(x)


def add_res_block(input, filters, kernel_size):
    mainres= MainRes(filters, kernel_size)
    out= mainres(input)
    res= input
    if out.shape[-1] != input.shape[-1]:
        res= Projection(out.shape[-1])(res)
    return tf.keras.layers.add([res, out])

class Model(tf.keras.Model):
    def __init__(self, height, width, frames, _class= 2, filters=16, kernel_size= (3,7,7), padding='same'):
        super().__init__()
        self.frames= frames
        self.height= height
        self.width= width
        self.conv3d= Conv3d(filters= filters, kernel_size=kernel_size, padding=padding)
        self.bn1= tf.keras.layers.BatchNormalization()
        self.relu= tf.keras.layers.ReLU()
        self.block1= MainRes(16, (3, 3, 3))
        self.block2= MainRes(32, (3, 3, 3))
        self.block3= MainRes(64, (3, 3, 3))
        self.block4= MainRes(128, (3, 3, 3))
        self.avgpool= tf.keras.layers.GlobalAveragePooling3D()
        self.flatten= tf.keras.layers.Flatten()
        self.dense= tf.keras.layers.Dense(_class)
        self.softmax= tf.keras.layers.Softmax()
    def resize(self, video, height, width):
        original= einops.parse_shape(video, 'b t h w c')
        images= einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images= tf.keras.layers.Resizing(height, width)(images)
        videos= einops.rearrange(images, '(b t) h w c -> b t h w c', t= original['t'])
        return videos

    def call(self, x):

        assert x.shape[1]==self.frames and x.shape[2]==self.height and x.shape[3]==self.width
        out= self.conv3d(x)
        out= self.bn1(out)
        out= self.relu(out)
        out= self.resize(out, self.height//2, self.width//2)
        out= self.block1(out)
        out= self.resize(out, self.height//4, self.width//4)
        out= self.block2(out)
        out= self.resize(out, self.height//8, self.width//8)
        out= self.block3(out)
        out= self.resize(out, self.height//16, self.width//16)
        out= self.block4(out)
        out= self.avgpool(out)
        out= self.flatten(out)
        out= self.dense(out)
        prob= self.softmax(out)
        return out, prob

    
