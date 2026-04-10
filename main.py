import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CREATE SYNTHETIC DATASET
# ==========================================
def create_synthetic_data(num_samples=25):
    images_day = []
    images_night = []
    
    for _ in range(num_samples):
        # Create "Day" image (Blue sky, green grass, yellow sun)
        day = np.zeros((256, 256, 3), dtype=np.float32)
        day[:160, :, :] = [0.4, 0.6, 1.0]  # Sky
        day[160:, :, :] = [0.1, 0.6, 0.1]  # Grass
        
        # Sun position (keeping it consistent with your sample image)
        sun_pos = [70, 190] 
        for i in range(256):
            for j in range(256):
                if (i-sun_pos[0])**2 + (j-sun_pos[1])**2 < 35**2:
                    day[i, j] = [1.0, 1.0, 0.0]
        
        # Create "Night" target (Dark sky, dark grass, grey moon)
        night = np.zeros((256, 256, 3), dtype=np.float32)
        night[:160, :, :] = [0.05, 0.05, 0.2] # Dark Sky
        night[160:, :, :] = [0.0, 0.2, 0.0]   # Dark Grass
        for i in range(256):
            for j in range(256):
                if (i-sun_pos[0])**2 + (j-sun_pos[1])**2 < 35**2:
                    night[i, j] = [0.8, 0.8, 0.8]
                    
        images_day.append((day * 2) - 1) # Normalize to [-1, 1]
        images_night.append((night * 2) - 1)
        
    return np.array(images_day), np.array(images_night)

# Load data
x_train, y_train = create_synthetic_data(25)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)

# ==========================================
# 2. THE PIX2PIX MODEL ENGINE
# ==========================================
def downsample(filters):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ])

def upsample(filters):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])

# Build Generator (U-Net style)
inputs = tf.keras.layers.Input(shape=[256, 256, 3])
d1 = downsample(64)(inputs)
d2 = downsample(128)(d1)
u1 = upsample(64)(d2)
concat = tf.keras.layers.Concatenate()([u1, d1])
last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(concat)

generator = tf.keras.Model(inputs=inputs, outputs=last)
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# ==========================================
# 3. RUN ONE TRAINING PASS (EPOCH)
# ==========================================
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as tape:
        gen_output = generator(input_image, training=True)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    grads = tape.gradient(l1_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

print("Training for 1 Epoch...")
# This loop runs exactly once through the 25 samples
for input_img, target_img in train_dataset:
    train_step(input_img, target_img)

# ==========================================
# 4. FINAL OUTPUT VISUALIZATION
# ==========================================
# Get a prediction on the first sample
prediction = generator(x_train[0:1], training=False)

plt.figure(figsize=(15, 7))

# MAIN TITLE
plt.suptitle("Image Translation : DAY → NIGHT", fontsize=22, fontweight='bold')

# First Column: Original Day
plt.subplot(1, 3, 1)
plt.title("Day Image", fontsize=14)
plt.imshow(x_train[0] * 0.5 + 0.5)
plt.axis('off')

# Second Column: AI Result
plt.subplot(1, 3, 2)
plt.title("Generated Night", fontsize=14)
plt.imshow(prediction[0].numpy() * 0.5 + 0.5)
plt.axis('off')

# Third Column: The Goal
plt.subplot(1, 3, 3)
plt.title("Target Night", fontsize=14)
plt.imshow(y_train[0] * 0.5 + 0.5)
plt.axis('off')

print("Process finished. Closing training...")
plt.show()
