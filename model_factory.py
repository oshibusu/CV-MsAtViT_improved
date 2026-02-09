import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from cvnn.layers import (
    complex_input,
    ComplexConv2D,
    ComplexConv3D,
    ComplexDense,
    ComplexDropout,
    ComplexFlatten,
    ComplexAvgPooling2D,
    ComplexBatchNormalization,
)
from SAR_utils import cart_gelu, num_classes, softmax_real_with_real, ComplexLayerNormalization, ModReLU, ModGated, ModSigmoidGated
from CoordAttention import CoordAtt_cmplx
from ComplexAttention import ComplexMultiHeadAttention


def cmplx_multilayer_perceptron(x, hidden_units, dropout_rate, activation_type="modrelu", b_init=-0.1):
    for units in hidden_units:
        if activation_type == "modrelu":
            x = ComplexDense(units, activation=None)(x)
            x = ModReLU(b_init=b_init)(x)
        elif activation_type == "mod_gated":
            x = ComplexDense(units, activation=None)(x)
            x = ModGated(b_init=b_init)(x)
        elif activation_type == "mod_sigmoid_gated":
            x = ComplexDense(units, activation=None)(x)
            x = ModSigmoidGated(b_init=b_init)(x)
        else:
            x = ComplexDense(units, activation=cart_gelu)(x)
        x = ComplexDropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = ComplexDense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + tf.cast(
            self.position_embedding(positions), tf.complex64
        )
        return encoded


def MultiScaleFeatureExtractor(inputs, activation_type="modrelu", b_init=-0.1):
    if activation_type in ["modrelu", "mod_gated", "mod_sigmoid_gated"]:
        act = None
    else:
        act = "cart_relu"

    x1 = ComplexConv3D(
        filters=8,
        kernel_size=(3, 3, 1),
        activation=act,
        padding="same",
        name="spatial_conv3d_block1",
    )(inputs)
    if activation_type == "modrelu":
        x1 = ModReLU(b_init=b_init)(x1)
    elif activation_type == "mod_gated":
        x1 = ModGated(b_init=b_init)(x1)
    elif activation_type == "mod_sigmoid_gated":
        x1 = ModSigmoidGated(b_init=b_init)(x1)

    x1 = ComplexConv3D(
        filters=8,
        kernel_size=(3, 3, 1),
        activation=act,
        padding="same",
        name="spatial_conv3d_block2",
    )(x1)
    if activation_type == "modrelu":
        x1 = ModReLU(b_init=b_init)(x1)
    elif activation_type == "mod_gated":
        x1 = ModGated(b_init=b_init)(x1)
    elif activation_type == "mod_sigmoid_gated":
        x1 = ModSigmoidGated(b_init=b_init)(x1)

    x2 = ComplexConv3D(
        filters=8,
        kernel_size=(1, 1, 3),
        activation=act,
        padding="same",
        name="polar_conv3d_block1",
    )(inputs)
    if activation_type == "modrelu":
        x2 = ModReLU(b_init=b_init)(x2)
    elif activation_type == "mod_gated":
        x2 = ModGated(b_init=b_init)(x2)
    elif activation_type == "mod_sigmoid_gated":
        x2 = ModSigmoidGated(b_init=b_init)(x2)

    x2 = ComplexConv3D(
        filters=8,
        kernel_size=(1, 1, 3),
        activation=act,
        padding="same",
        name="polar_conv3d_block2",
    )(x2)
    if activation_type == "modrelu":
        x2 = ModReLU(b_init=b_init)(x2)
    elif activation_type == "mod_gated":
        x2 = ModGated(b_init=b_init)(x2)
    elif activation_type == "mod_sigmoid_gated":
        x2 = ModSigmoidGated(b_init=b_init)(x2)

    x3 = ComplexConv3D(
        filters=8,
        kernel_size=(3, 3, 3),
        activation=act,
        padding="same",
        name="joint_conv3d_block1",
    )(inputs)
    if activation_type == "modrelu":
        x3 = ModReLU(b_init=b_init)(x3)
    elif activation_type == "mod_gated":
        x3 = ModGated(b_init=b_init)(x3)
    elif activation_type == "mod_sigmoid_gated":
        x3 = ModSigmoidGated(b_init=b_init)(x3)

    x3 = ComplexConv3D(
        filters=8,
        kernel_size=(3, 3, 3),
        activation=act,
        padding="same",
        name="joint_conv3d_block2",
    )(x3)
    if activation_type == "modrelu":
        x3 = ModReLU(b_init=b_init)(x3)
    elif activation_type == "mod_gated":
        x3 = ModGated(b_init=b_init)(x3)
    elif activation_type == "mod_sigmoid_gated":
        x3 = ModSigmoidGated(b_init=b_init)(x3)

    concatenated_features = tf.concat([x1, x2, x3], axis=4)
    return concatenated_features


def cmplx_ViT(
    x,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
    layer_norm_type="amplitude",
    activation_type="modrelu",
    b_init=-0.1,
):
    patches = Patches(patch_size)(x)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        if layer_norm_type == "complex":
            x1 = ComplexLayerNormalization(epsilon=1e-6)(encoded_patches)
        elif layer_norm_type == "amplitude":
            # Keeping 'amplitude' as an option if needed, but implementation is replaced by ComplexLayerNormalization in this refactor plan?
            # User request: "Overwrite report_20260126.md with Complex Layer Normalization plan" implied replacing the logic.
            # But here "layer_norm_type='complex'" matches the plan.
            # Let's map "complex" to ComplexLayerNormalization.
            x1 = ComplexLayerNormalization(epsilon=1e-6)(encoded_patches)
        else:
            x1_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(encoded_patches))
            x1_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(encoded_patches))
            x1 = tf.complex(x1_r, x1_i)
        
        attention_output = ComplexMultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        if layer_norm_type == "complex":
            x3 = ComplexLayerNormalization(epsilon=1e-6)(x2)
        elif layer_norm_type == "amplitude":
             x3 = ComplexLayerNormalization(epsilon=1e-6)(x2)
        else:
            x3_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(x2))
            x3_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(x2))
            x3 = tf.complex(x3_r, x3_i)

        x3 = cmplx_multilayer_perceptron(
            x3, hidden_units=transformer_units, dropout_rate=0.1, activation_type=activation_type, b_init=b_init
        )

        encoded_patches = layers.Add()([x3, x2])

    
    if layer_norm_type == "complex" or layer_norm_type == "amplitude":
        representation = ComplexLayerNormalization(epsilon=1e-6)(encoded_patches)
    else:
        representation_r = layers.LayerNormalization(epsilon=1e-6)(
            tf.math.real(encoded_patches)
        )
        representation_i = layers.LayerNormalization(epsilon=1e-6)(
            tf.math.imag(encoded_patches)
        )
        representation = tf.complex(representation_r, representation_i)

    representation = ComplexFlatten()(representation)
    representation = ComplexDropout(0.5)(representation)

    features = cmplx_multilayer_perceptron(
        representation, hidden_units=mlp_head_units, dropout_rate=0.3, activation_type=activation_type, b_init=b_init
    )

    return features


def build_msatvit(
    input_shape,
    dataset,
    window_size=15,
    lr=None,
    patch_size=3,
    projection_dim=32,
    num_heads=4,
    transformer_layers=4,
    mlp_head_units=None,
    transformer_units=None,
    layer_norm_type="complex",
    activation_type="modrelu",
    coord_activation="modtanh",
    b_init=-0.1,
):
    if lr is None:
        lr = 0.0001 if dataset == "ober" else 0.001
    transformer_units = transformer_units or [projection_dim * 2, projection_dim]
    mlp_head_units = mlp_head_units or [1024, 512]
    num_patches = (window_size // patch_size) ** 2

    inputs = complex_input(shape=input_shape)
    x = MultiScaleFeatureExtractor(inputs, activation_type=activation_type, b_init=b_init)
    x_shape = x.shape
    x = layers.Reshape((x_shape[1], x_shape[2], x_shape[3] * x_shape[4]))(x)
    
    act_conv = None if activation_type in ["modrelu", "mod_gated", "mod_sigmoid_gated"] else "cart_relu"
    x = ComplexConv2D(filters=24, kernel_size=(3, 3), activation=act_conv, padding="same")(x)
    if activation_type == "modrelu":
        x = ModReLU(b_init=b_init)(x)
    elif activation_type == "mod_gated":
        x = ModGated(b_init=b_init)(x)
    elif activation_type == "mod_sigmoid_gated":
        x = ModSigmoidGated(b_init=b_init)(x)

    x = CoordAtt_cmplx(x, 4, activation=coord_activation)
    x = cmplx_ViT(
        x,
        patch_size,
        num_patches,
        projection_dim,
        num_heads,
        transformer_units,
        transformer_layers,
        mlp_head_units,
        layer_norm_type=layer_norm_type,
        activation_type=activation_type,
        b_init=b_init,
    )
    z = ComplexFlatten()(x)
    logits = ComplexDense(num_classes(dataset), activation=softmax_real_with_real)(z)

    model = tf.keras.Model(inputs=[inputs], outputs=logits)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


from cvnn_fix import FixedComplexBatchNormalization
from CoordAttention import CoordAtt_cmplx, ComplexSplit

CUSTOM_OBJECTS = {
    "ComplexConv2D": ComplexConv2D,
    "ComplexConv3D": ComplexConv3D,
    "ComplexDense": ComplexDense,
    "ComplexDropout": ComplexDropout,
    "ComplexFlatten": ComplexFlatten,
    "ComplexAvgPooling2D": ComplexAvgPooling2D,
    "ComplexBatchNormalization": ComplexBatchNormalization,
    "cart_gelu": cart_gelu,
    "CoordAtt_cmplx": CoordAtt_cmplx,
    "ComplexSplit": ComplexSplit,
    "ComplexMultiHeadAttention": ComplexMultiHeadAttention,
    "softmax_real_with_real": softmax_real_with_real,
    "ComplexLayerNormalization": ComplexLayerNormalization,
    "ModReLU": ModReLU,
    "ModGated": ModGated,
    "ModSigmoidGated": ModSigmoidGated,
}

# Use FixedComplexBatchNormalization only when loading the SavedModel to bypass the TypeError
LOADING_CUSTOM_OBJECTS = CUSTOM_OBJECTS.copy()
LOADING_CUSTOM_OBJECTS["ComplexBatchNormalization"] = FixedComplexBatchNormalization

def load_saved_msatvit(saved_model_dir: str):
    if not tf.io.gfile.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")
    return tf.keras.models.load_model(
        saved_model_dir,
        compile=False,
        custom_objects=LOADING_CUSTOM_OBJECTS,
    )
