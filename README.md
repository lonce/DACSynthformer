## DACSynthformer

DACSynthformer is a basic transformer that runs on the Descript Audio Codec representation of audio. It maintains the "stacked" codebook at each transformer time step (as oppose to laying the codebooks down "horizonally" for examaple). It uses a smallish causal mask during training, so that during autoregressive inference we can use a small context window. It uses RoPE positional encoding because absolute positions are irrelevant for the continuous and stable audio textures we wish to generate. Conditioning is provided as a vector combining a one-hot segment for sound class, and real number(s) for parameters.

## Install option 1 (conda only) 

This option is perfect for CPU's. You need to have miniconda (or conda) installed: https://docs.anaconda.com/miniconda/install/ . Then:

~~~
conda create --name synthformer python==3.10
conda activate synthformer
pip install -r requirements.txt
jupyter lab &
~~~

(If you use Windows, I suggest you use a command window and not a powershell window. Sheesh.)

## Install option 2 (Docker) 

> ### Create the docker container  
> ~~~
> #updated Feb 8, 2025 to use docker buildx:
> docker buildx build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile.txt --tag yourtag --load .
> ~~~

> ### Run the docker container
> ~~~
> docker run --ipc=host --gpus all -it -v $(pwd):/dacsynthformer  -v /home/lonce/scratchdata:/scratch --name callitwhatyouwill --rm -p 8888:8888  yourtag
> cd /dactransformer
> jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
> ~~~
> I use scratch as the root directory for data, etc. 

## There are 2 primary notebooks:  
1) Train.ipynb - this is the main notebook for training the model. The dataloader loads pre-coded DAC files (4 codebooks for 44.1kHz sampled audio). It creates checkpoints that you can use to generate audio. 
2) Inference.Decode.ipynb - uses a stored trained model to first generate a DAC coded file and then decodes that to audio with the Descript codec ("DAC"). 

Each of the notebooks has a "**Parameters**" section near the top.

* In Train.ipynb, choose your params.yaml file (which is where most of your model parameters are set). The  "**Parameters**" section also is where you choose the platform ('cpu' or 'cuda') and the starting epoch number (0 unless you want to pick up training where a previous run left off). 

* In Inference.Decode.ipynb, the "**Parameters**" is where you set the "experiment name" to whatever you named your experiment in the params.yaml file you trained with. 

  

## Quick Start:

Once you have created and are in your conda (or docker) environment and have started jupyter, just:

* Open Train.ipynb and set paramfile = 'params_mini.yaml' in the "**Parameters**" section.
* Run All Cells (You can see that params_mini.yaml uses testdata/ with 4 sounds, and trains on 10 epochs.)
* Then open Inference.Decode.ipynb and set  experiment_name="mini_test_01" in the "**Parameters**" section of , and cptnum=10.  
* Run All Cells 

This runs a very minimal model, and doesn't train long enough to generate anything but codec noise. The intention is that you can see that Training and Inference code is functioning! When that works, you are ready to start creating a new dataset and exploring the model!

**Note:** the parameter you choose for 'device' must be available on your system, and can be either 'cpu', 'cuda', or (for Macs with "Metal Performance Shaders") 'mps'.



## Preparing data: 


The conda and docker environments have already installed the Descript DAC codec package, so you can encode .wav files to .dacs and decode .dac file to .wavs. Just encode a folder of wav files like this:

`python3 -m dac encode /my/wavs/folder --model_bitrate 8kbps --n_quantizers 4 --device cpu --output my/output/folder/.` 

The --device you specify needs to be available, and it's default value is 'cuda'. For more information about the Descript codec, see the README here:

https://github.com/descriptinc/descript-audio-codec

**Note:** All files must be the same length (have the same number of dac frames) and that length will be one more than the context window used for training (becuase the target is the input shifted by 1). (I use 5 seconds of audio which the encode command converts to 431 dac frames (86 frames/sec). The Tt variable ("**T**imesteps **t**raining" = context length) must be set in your param yaml file, too (one less than your file length). You can check the sequence length of your files: 

```bash
python    sfutils/dacFileSeqLength.py    path/to/foo.dac
```



**Create xlsx spred of data frames for pandas:**

Then prepare your excel data file (that pandas will use). It should have columns, with labels in the first row:

Full File Name     |        Class Name         |    Param1   | ....  | ParamN

The file name includes no path (you provide that in a params.yaml config file). Class Names are whatever you choose. Synthformer will create a separate one-hot class element in the conditioning vector used for training and inference for each unique Class Name. (You can see examples of the excel files in testdata). Consider a balance of classes for your training data!  The column headers for the params can be whatever name you choose to give them. 

Note: you typically write a little python program to generate your excel Pandas Frames from your collection of data files. I generally create datafiles with names that can be parsed to create the excel frames. 



## To train:  

1) Edit (or create) a parameter.yaml file with model parameters, folder names for your data and file names for the excel Pandas file.  
2) Open Train.ipynb, set your parameter.yaml file  in the "**Parameters**" section of Training.ipynb and any other parameters you want (such as  DEVICE). 
3) Run all cells.
4) There is a ridiculously tiny data set of dac files in test/data, and a few prepared parameter files:
   1)  params_mini.yaml - for QuickStart and code debugging. Runs all code, but with minimal data and a tiny model.
   2)  params_sm.yaml - good for debugging since it quickly uses all stages of the computation.
   3)  params_med.yaml - uses a slightly bigger model, and the tiny data set for training period long enough to actually see that training and inference work. This can also run on a CPU.
   4)  params.yaml - defines a bigger model, and meant for running a larger dataset for many epochs.



Loss is printed to the notebook output window, and to your output directory so that they can be watched using tensorboard (pip install tensorboard), by running:

`tensorboard --logdir=./runs/your_experiment_name  --port=6006`

then point your browser to localhost:6006

## Optional Info: 

* Prepared dataset:

The testdata/ folder has everything you need as a template for creating your own dataset. However, here is a dataset along with a param.yaml file that specifies a medium-size model that you can use for testing, seeing how bi you might want your data and model to be for your own work, etc: 

https://drive.google.com/file/d/1IdMb4v9wD4nHlFLFJe-pl85rFQW0eF-Y/view?usp=sharing It seems you actually have to go to this google drive page and click the download button. 

With this data and model spec, I see training at a rate of about 2 minutes per epoch on a powerful desktop machine (CPU only), and about 7 minutes/epoch on my PC. You can see that it is training after about just 5 epochs (That's 35 minutes on my laptop!), and starting to produce something reasonable for some of the sounds after 20 epochs. Reduce the size of the model for speedier training. Your datasets can be much smaller, though - try just 2 or 3 classes. 

The sounds are from the Syntex sound textures data set syntex data set ( https://syntex.sonicthings.org/soundlist) 

* Pistons, with a 'rate' parameter,
*  Wind, with a 'gustiness' parameter
* Applause, with a 'number of clappers' parameter
* Bees (bugs , with a 'busy-body' parameter (how fast and far the bees move)
* Peepers, with a 'frequency range' parameter
* TokWottle, with a 'wood to metal' hit ratio parameter 
* FM, with a 'modulation frequency' parameter

These sounds all sample their parameter space [0,1] in steps of .05. There are 630 files per sound class (30 5 second samples at each parameter value), totaling about 6 hours of audio. 

<div style="background-color: lightblue; padding: 10px;">
</div>

## Running on mel spectrograms instead of DAC: 

The Synthformer can be run on Mel spectrograms, too. Although this means regressing to predict spectrograms rather than probabilities over a vocabulary, most of the core transformer code is the same. 

In your activated synthformer conda environment, clone BigVGAN: https://github.com/NVIDIA/BigVGAN . You should then have the BigVGAN folder in your root folder. cd to your BigVGAN folder, and run

```
pip install -r requirements.txt
```

and then install (one of) their pretrained models:

```shell
git lfs install
git clone https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x
```

 change directory back up to your root, and add BigVGAN to your PYTHONPATH env variable:  
```shell
export PYTHONPATH="$(pwd):$(pwd)/BigVGAN:$PYTHONPATH" 
```

Now you are ready to convert a directory of wav files (44.1kHz, mono, all the same length) to mel spectrogram which will be written as .npy files: 
```shell
python shutils/wavs2mels.py /path/to/wavefiles  /path/to/melfiles
```

Next create the data frames file (see **Create xlsx spred of data frames for pandas:** above). Also see  ./testdata/Lala_data for what the files and file structure might look like. 
