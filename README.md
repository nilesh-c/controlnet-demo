
# ControlNet Demo API

This repository provides a Dockerized API that uses ControlNet to perform image generation tasks. The API allows you to interact with the ControlNet+StableDiffusion 1.5 model for generating images conditioned on a given image and prompts using a simple FastAPI-based service. This project is designed to be used with NVIDIA GPUs and Docker for easy setup and execution. The `/generate_mri` endpoint allows you to directly generate synthetic brain MRI images.



## Project Structure

```plaintext
|-requirements.txt
|-test_imgs/
| |-mri_brain.jpg
|-demo/
| |-model.py
| |-generate_image.py
| |-test_client.ipynb
| |-app.py
|-docker-compose.yml
|-Dockerfile
|-models/
| |-cldm_v15.yaml
|-README.md
|-LICENSE
```

- `requirements.txt`: Python dependencies required for the project.
- `test_imgs/`: Folder containing test images (e.g., `mri_brain.jpg`) used for image generation.
- `demo/`: Contains Python files that interact with the model, including `model.py`, `generate_image.py`, and `app.py` for the FastAPI application.
- `models/`: Folder to store model weights like `control_sd15_canny.pth` (needs to be downloaded from HuggingFace).
- `Dockerfile`: Dockerfile to build the environment with necessary dependencies.
- `docker-compose.yml`: Docker Compose configuration to build and run the API in a container.
- `README.md`: Documentation for setting up and running the project.

## Prerequisites

Before using this project, you need:

- Docker and Docker Compose installed on your machine (if running with Docker).
- Miniconda installed if running with Conda.
- NVIDIA GPU with the `nvidia-container-toolkit` installed (only needed if using Docker) for GPU acceleration.

## Setup Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/nilesh-c/controlnet-demo-api.git
cd controlnet-demo-api
```

### Step 2: Download Model Weights

You need to download the model `control_sd15_canny.pth` from HuggingFace into the `models/` directory. Use the `huggingface-cli` to download the model:

```bash
huggingface-cli download lllyasviel/ControlNet models/control_sd15_canny.pth --local-dir ./
```

Alternatively, you can manually download the model and place it in the `models/` directory.

### Step 3: Running the Application

#### Option 1: Running with Conda

1. **Clone ControlNet and create Conda environment:**
   ```bash
   git clone https://github.com/lllyasviel/ControlNet.git
   cd ControlNet
   conda env create -f environment.yaml
   conda activate control
   cd ..
   ```

2. **Install demo repo dependencies:**

   Run the following command to install the dependencies required for running the REST API demo:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI application:**

   Once all dependencies are installed, start the FastAPI app by running:

   ```bash
   uvicorn demo.app:app --host 0.0.0.0 --port 8000
   ```

   This will start the application at `http://localhost:8000`. You can test the API using the provided Jupyter notebook (`test_client.ipynb`) or make HTTP requests directly.

#### Option 2: Running with Docker

1. **Build the Docker image** using Docker Compose:

   ```bash
   docker-compose build
   ```

2. **Start the container** with the following command:

   ```bash
   docker-compose up
   ```

   This will start the FastAPI application in a Docker container. The API will be accessible on port `8000`.

### Step 4: Test the API

Once the container is running (or the Conda environment is activated), you can interact with the API.

#### Using Jupyter Notebook (`test_client.ipynb`)

The Jupyter notebook contains examples of how to interact with the API. Open the notebook in your browser or use a Jupyter environment to run the code. This notebook demonstrates how to send images to the API and receive generated images in response. The `/generate_mri` endpoint allows you to directly generate synthetic brain MRI images.

#### Using HTTP Requests

The API is exposed on port `8000`. You can use tools like `curl`, Postman, or your browser to interact with the API. Here is an example of how to make a request to the API:

```bash
curl -X POST "http://localhost:8000/generate" -F "file=@test_imgs/mri_brain.jpg" -F "prompt=mri brain scan" -F "a_prompt=good quality" -F "n_prompt=animal, drawing, painting"
```

This will send an image to the API and return the generated image.

Alternatively, you can use the `/generate_mri` endpoint to generate a default image from a pre-defined MRI brain scan image:

```bash
curl -X POST "http://localhost:8000/generate_mri"
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

### Additional Notes:

- Ensure that you have a working GPU setup and that the Docker container is running with NVIDIA GPU support. Consult https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- The model weights (`control_sd15_canny.pth`) are required to perform image generation, so make sure you download them before running the application.
- The `generate_mri` endpoint uses a hardcoded MRI brain scan image, but you can customize this as needed.
