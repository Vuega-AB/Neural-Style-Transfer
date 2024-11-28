import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.pyplot import imshow
from PIL import Image
import pprint
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(272) 
pp = pprint.PrettyPrinter(indent=4)
# Image size
img_size = 400

# Load VGG19 model without the top layer
vgg = tf.keras.applications.VGG19(include_top=False, 
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False
pp.pprint(vgg)

# Streamlit title
st.title("Neural Style Transfer")
st.write("Upload content and style images to apply neural style transfer.")

# Streamlit image upload
content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# Check if both images are uploaded
if content_image_file and style_image_file:
    content_image = Image.open(content_image_file)
    content_image= content_image.convert("RGB")
    content_image = np.array(content_image.resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))


    style_image = Image.open(style_image_file)
    style_image = style_image.convert("RGB")
    style_image = np.array(style_image.resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    content_image_np = np.array(content_image[0])  # Convert the first image in the batch
    style_image_np = np.array(style_image[0])  # Convert the first image in the batch

    # Display the uploaded images
    st.image(content_image_np, caption="Content Image", use_column_width=True)
    st.image(style_image_np, caption="Style Image", use_column_width=True)




    # Generate a random noise image to start with
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    # Model outputs for style and content layers
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)
    ]
    content_layer = [('block5_conv4', 1)]

    # Helper function to get the VGG19 model outputs
    def get_layer_outputs(vgg, layer_names):
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
        return tf.keras.Model([vgg.input], outputs)

    # Get the output of the content and style layers
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs(style_image)     # Style encoder

    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    # Define the content and style cost functions
    def compute_content_cost(content_output, generated_output):
        a_C = content_output[-1]
        a_G = generated_output[-1]
        _, n_H, n_W, n_C = a_G.get_shape().as_list()
        a_C_unrolled = tf.reshape(a_C, shape=[1, n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[1, n_H * n_W, n_C])
        J_content = (1/(4 * n_H * n_W * n_C)) * tf.reduce_sum((a_C - a_G)**2)
        return J_content

    def gram_matrix(A):
        return tf.linalg.matmul(A, tf.transpose(A))

    def compute_layer_style_cost(a_S, a_G):
        _, n_H, n_W, n_C = a_G.get_shape().as_list()
        a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        J_style_layer = (1/(4 * (n_H * n_W * n_C)**2)) * tf.reduce_sum((GS - GG)**2)
        return J_style_layer

    def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
        J_style = 0
        a_S = style_image_output[:-1]
        a_G = generated_image_output[:-1]
        for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
            J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
            J_style += weight[1] * J_style_layer
        return J_style

    def total_cost(J_content, J_style, alpha=10, beta=40):
        return alpha * J_content + beta * J_style

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            tensor = tensor[0]
        return Image.fromarray(tensor)

    # Training step with gradient descent
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function()
    def train_step(generated_image):
        with tf.GradientTape() as tape:
            a_G = vgg_model_outputs(generated_image)
            J_style = compute_style_cost(a_S, a_G)
            J_content = compute_content_cost(a_C, a_G)
            J = total_cost(J_content, J_style, alpha=10, beta=40)
        grad = tape.gradient(J, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip_0_1(generated_image))
        return J

    generated_image = tf.Variable(generated_image)

    # Run the style transfer for a number of epochs
    epochs = 500
    for i in range(epochs):
        train_step(generated_image)
        if i % 250 == 0:
            print(f"Epoch {i} completed")
            if i % 250 == 0:
                image = tensor_to_image(generated_image)
                st.image(image, caption=f"Generated Image at Epoch {i}", use_column_width=True)
        
    # Display the final generated image
    st.write("Final Generated Image:")
    generated_image_np = np.array(generated_image[0])  # Convert the first image in the batch
    generated_image_pil = tensor_to_image(generated_image)  # Convert tensor to PIL for final output

    # Display the final generated image
    st.image(generated_image_pil, caption="Generated Image", use_column_width=True)


    # Display images in a row (content, style, generated)
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    imshow(content_image[0])
    ax.title.set_text('Content Image')
    ax = fig.add_subplot(1, 3, 2)
    imshow(style_image[0])
    ax.title.set_text('Style Image')
    ax = fig.add_subplot(1, 3, 3)
    imshow(generated_image[0])
    ax.title.set_text('Generated Image')
    plt.show()
