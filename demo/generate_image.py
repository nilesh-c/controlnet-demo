import imageio
from model import BrainMRIImageGenerator

import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Initializing BrainMRIImageGenerator...")

    # Initialize model
    generator = BrainMRIImageGenerator()

    # Load original image to get edge map
    input_image_path = "test_imgs/mri_brain.jpg"
    print(f"Loading test image: {input_image_path}")
    input_image = imageio.imread(input_image_path)

    # Generate output images
    print("Generating MRI scan image...")
    prompt = "mri brain scan"
    a_prompt = 'good quality' # 'best quality, extremely detailed'
    n_prompt = 'animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    results = generator.generate_image(input_image,
                                    prompt=prompt,
                                    a_prompt=a_prompt,
                                    n_prompt=n_prompt,
                                    apply_luminance=True)

    # Save and display results
    output_image_path = "output.png"
    imageio.imwrite(output_image_path, results[-1])
    print(f"Generated image saved as: {output_image_path}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_image)
    axs[0].set_title("Input Image")
    axs[1].imshow(255 - results[0])
    axs[1].set_title("Edge Detection")
    axs[2].imshow(results[-1])
    axs[2].set_title("Generated Image")

    for ax in axs:
        ax.axis("off")

    plt.show()