# Flow matching: Novel View Synthesis and Pose Estimation

In this project Jimmy and I have implemented a Machine Learning model based on the flow matching technique to create novel scenes and estimate the pose of the camera. In total we have done the following:

- We have created a custom dataset using a NeRF to enrich our already existing data with more images.
- We have trained an SiT (Scalable interpolant Transformer) to generate new scenes given a camera pose
- We have reversed the SiT to estimate the camera pose given an image scene.

## What has been modified?

We have adapted the original SiT repository to perform the following:

- Adapt the training script `flow_matching/train.py` for single class.
- Adapt the training script so that it ingest directly the numpy arrays for training instead of jpg files through `flow_matching/dataset.py`.
- Shift the gaussian mean to poses instead of mean 0, so that we predict new scenes given our poses.
- Implement bf16 training to save CUDA memory.
- Separate the evaluation from the training loop to prevent CUDA out of memory.
- Implement evaluation of the trained model `flow_matching/notebooks/evaluate.ipynb`.
- Create turntable dataset for model evaluation.
- Implement notebook for interpolation experiment `flow_matching/notebooks/interpolation_experiment.ipynb` with the trained model.
- Implement flow reversal. See "Test Invertibility of SiT" part in `flow_matching/notebooks/evaluate.ipynb` and "Pose Recovery and Error Evaluation" part in `flow_matching/notebooks/scene_prediction_loop.ipynb`.

## Setup

Step 1: Download and set up the repo:

```bash
git clone https://github.com/amndzdzdz/novel-view-synthesis-pose-estimation.git
cd novel-view-synthesis-pose-estimation
```

Step 2: Create a conda environment:

```bash
conda env create -f environment.yml
conda activate novel-view-synthesis-pose-estimation
```

If you only want to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the environment.yml file.

## Training SiT

Run the following command (for more details, check the original SiT repo!)

```bash
torchrun --nnodes=1 --nproc_per_node=1 flow_matching/train.py --model SiT-B/8 --num-classes 1 --epochs 300 --global-batch-size 150 --num-workers 0 --ckpt-every 2000 --cfg-scale 1.0 --image-size 128
```

`Note`: The dataset is generated using the code provided in the `nerf` folder

## Test your model after training

You can use the trained model to generate images using the poses with the `flow_matching/inference.py` script. It will output both the generated images as well as the ground truth images for comparison to the specified directory.

## Model evaluation using several metrics

To evaluate our model, we have implemented evaluations using several metrics.

The notebook `flow_matching/notebooks/evaluate.ipynb` evaluates both the image generation and pose estimation.

The notebook `flow_matching/notebooks/scene_prediction_loop.ipynb` test the model further by:

1. First View synthesis: Predict scene given noisy pose
2. Pose Estimation: Estimate the pose given the new scene
3. Second View synthesis: Predict scene again with estimated pose

## Additional materials

We have included our report on the project in `report.pdf`

## Contributors (equal contributions)

- Jimmy Tan
- Amin Dziri
