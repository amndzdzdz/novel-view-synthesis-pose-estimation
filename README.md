# This repository is modified based on the original [SiT repository](https://github.com/willisma/SiT). 
This is the repository of the Image and Video Synthesis SoSe 2025 Practical for Pose Estimation + Novel View Synthesis. 

## What has been modified? 
We have adapted the original SiT repository to perform the following:

- [x] Adapt the training script (train.py) for single class.  
- [x] Adapt the training script so that it ingest directly the numpy arrays for training instead of jpg files (using dataset.py). 
- [x] Shift the gaussian mean to poses instead of mean 0. 
- [x] Implement bf16 training to save CUDA memory. 
- [x] Separate the evaluation from the training loop to prevent CUDA out of memory. 
- [x] Implement evaluation of the trained model (evaluate.ipynb). 
- [x] Create turntable dataset for model evaluation (refer to [this repository](https://github.com/JimmyTan2000/nerf-image-generation)). 
- [x] Implement notebook for interpolation experiment (interpolation_experiment.ipynb) with the trained model. 
- [x] Implement flow reversal. See "Test Invertibility of SiT" part in `evaluate.ipynb` and "Pose Recovery and Error Evaluation" part in `scene_prediction_loop.ipynb`.


## Setup
Step 1: Download and set up the repo:
```bash 
git clone https://github.com/JimmyTan2000/SiT.git
cd SiT
```
Step 2: Create a conda environment:
```bash
conda env create -f environment.yml
conda activate SiT
```
`Note`: We have added some more packages during the course of development. If the above environment is incomplete, try this instead:
```bash
conda env create -f environment_full.yml
conda activate SiT
```
If you only want to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the environment.yml file.

## Training SiT

Run the following command (for a single GPU setup)
```bash 
torchrun --nnodes=1 --nproc_per_node=1 train.py --model SiT-B/8 --num-classes 1 --epochs 300 --global-batch-size 150 --num-workers 0 --ckpt-every 2000 --cfg-scale 1.0 --image-size 128
```
`Note`: Our dataset is generated using [this repository](https://github.com/JimmyTan2000/nerf-image-generation).

## Test your model after training
You can use the trained model to generate images using the poses with the `model_inference.py` script. It will output both the generated images as well as the ground truth images for comparison to the specified directory.

### Alternative testing (depreciated)
Before the implementation of `model_inference.py` script, the ema sampling generation in `train.py` is moved out to a separate script `generate_samples.py`. We have commented out the ema checkpoint saving code in the `train.py`. Stated below is the commented-out code: 
```python
if train_steps % args.sample_every == 0 and train_steps > 0:
    if rank == 0:
        ema_sampling_dir = os.path.join(experiment_dir, "ema_sampling_checkpoints")
        os.makedirs(ema_sampling_dir, exist_ok=True)
        ema_ckpt = {
            "ema": ema.state_dict(),
            "step": train_steps,
            "args": args,
        }
        ema_ckpt_path = os.path.join(ema_sampling_dir, f"ema_sampling_{train_steps:07d}.pt")
        torch.save(ema_ckpt, ema_ckpt_path)
        logger.info(f"Saved EMA checkpoint for sampling at step {train_steps}")
    dist.barrier()
``` 
We recommend just using `model_inference.py` for the test. `generate_samples.py` is depreciated. To use it, you must uncomment the above code. 

## Model evaluation using several metrics
To evaluate our model, we have implemented evaluations using several metrics. 

The notebook `evaluate.ipynb` evaluates both the image generation and pose estimation. 

The notebook `scene_prediction_loop.ipynb` test the model further by: 

1. First View synthesis: Predict scene given noisy pose
2. Pose Estimation: Estimate the pose given the new scene
3. Second View synthesis: Predict scene again with estimated pose

## Additional materials
We have included our presentation slide `Practical Image and Video Synthesis Presentation.pdf`, as well as our report `Practical_Image_and_Video_Synthesis_Report.pdf` in our repository, hoping to clarify potential questions. 

## Contributors (equal contributions)
- Jimmy Tan 
- Amin Dziri