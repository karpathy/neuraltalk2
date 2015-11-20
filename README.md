
# NeuralTalk2

Recurrent Neural Network captions your images. Now much faster and better than the original [NeuralTalk](https://github.com/karpathy/neuraltalk). This implementation is **batched, uses Torch and runs on a GPU**. This results in about ~200x increase in training speed. It turns out that this also translates to much better performance.

This is an early code release that works great but is slightly hastily released and probably requires some code reading of inline comments (which I tried to be quite good with in general). I will be improving it over time but wanted to push the code out there because I promised it to too many people.

This current code (and the pretrained model) gets about 0.9 CIDEr, which would place it around spot #8 on the [codalab leaderboard](https://competitions.codalab.org/competitions/3221#results). I will submit the result soon.

![teaser results](https://raw.github.com/karpathy/neuraltalk2/master/vis/teaser.jpeg)

### Requirements

This code is written in Lua and requires [Torch](http://torch.ch/). If you're on Ubuntu, installing Torch in your home directory may look something like: 

```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
$ luarocks install nn
$ luarocks install nngraph 
$ luarocks install image 
```

If you'd like to train on an NVIDIA GPU using CUDA (which you REALLY REALLY want to, since we're using a VGGNet), you'll of course need a GPU, and you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then get the `cutorch` and `cunn` packages:

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

If you'd like to train your models you will need [loadcaffe](https://github.com/szagoruyko/loadcaffe), since we are using the VGGNet.

```bash
luarocks install loadcaffe
```

Lastly, if you want to train you will also need to install [torch-hdf5](https://github.com/deepmind/torch-hdf5), and [h5py](http://www.h5py.org/), since we will be using hdf5 files to store the preprocessed data.

### I just want to caption images

In this case you want to run the evaluation script on a pretrained model checkpoint. 
I trained a decent one on the [MS COCO dataset](http://mscoco.org/) that you can run on your images.
The pretrained checkpoint can be downloaded here: [pretrained checkpoint link](http://cs.stanford.edu/people/karpathy/neuraltalk2/checkpoint_v1.zip) (600MB). It's large because it contains the weights of a finetuned VGGNet. Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ th eval.lua -model /path/to/model -image_folder /path/to/image/directory
```

The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:4000` in your browser and you should see your predicted captions.

You can see an [example visualization demo page here](http://cs.stanford.edu/people/karpathy/neuraltalk2/demo.html).

### I'd like to train my own network on MS COCO

Great, first we need to some preprocessing. Head over to the `coco/` folder and run the IPython notebook to download the dataset and do some very simple preprocessing. The notebook will combine the train/val data together and create a very simple and small json file that contains a large list of image paths, and raw captions for each image, of the form:

```
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
```

Once we have this, we're ready to invoke the `prepro.py` script, which will read all of this in and create a dataset (an hdf5 file and a json file) ready for consumption in the Lua code. For example, for MS COCO we can run the prepro file as follows:

```bash
$ python prepro.py --input_json coco/coco_raw.json --num_val 5000 --num_test 5000 --images_root coco/images --word_count_threshold 5 --output_json coco/cocotalk.json --output_h5 coco/cocotalk.h5
```

This is telling the script to read in all the data (the images and the captions), allocate 5000 images for val/test splits respectively, and map all words that occur <= 5 times to a special `UNK` token. The resulting `json` and `h5` files are about 30GB and contain everything we want to know about the dataset. The last thing we need is the [VGG-16 caffe checkpoint](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), (under Models section, "16-layer model" bullet point).

We're now ready to train!

```bash
$ th train.lua -input_h5 coco/cocotalk.h5 -input_json coco/cocotalk.json
```

The train script will take over, and start dumping checkpoints into the folder specified by `checkpoint_path` (default = current folder). You also have to point the train script to the VGGNet protos (see the options inside `train.lua`).

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training, turn on `language_eval` option, but don't forget to download the [cococaption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

### I'd like to train on my own data

No problem, create a json file in the exact same form as before:

```
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
```

and invoke the `prepro.py` script to preprocess all the images and data into and hdf5 file. Then train the model with `train.lua` (see docs inside code).

### Acknowledgements

Parts of this code were written in collaboration with my labmate [Justin Johnson](http://cs.stanford.edu/people/jcjohns/). 

We're very grateful for [NVIDIA](https://developer.nvidia.com/deep-learning)'s support in providing GPUs that made this work possible.
