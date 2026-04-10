## 📂 Task-04: Image-to-Image Translation with cGAN

### Description
This task utilizes a **Conditional Generative Adversarial Network (cGAN)**, specifically the **pix2pix** architecture, to perform image translation. The model learns the mapping from an input image to an output image, demonstrated here through a "Day to Night" transformation.

### Features
 * **Conditional Generation:** Uses GANs to generate specific outputs based on an input image.
 * **Day \rightarrow Night Mapping:** Translates sunlight, sky color, and environmental lighting into a nighttime aesthetic.
 * **Validation:** Comparison between "Generated Night" and "Target Night" (Ground Truth) to evaluate model performance.
   
### Tech Stack
 * Python
 * TensorFlow / Keras
 * Matplotlib (for result visualization)
 * OpenCV
 * Custom GUI Framework
