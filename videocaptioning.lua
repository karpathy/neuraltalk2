require 'torch'
require 'nn'
require 'nngraph'
-- exotics
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgcodecs'
require 'cv.imgproc'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','','path to model to evaluate')
-- Basic options
cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', 100, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 1, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 1, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

cv.namedWindow{winname="NeuralTalk2", flags=cv.WINDOW_AUTOSIZE}
local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
  print("Failed to open the default camera")
  os.exit(-1)
end
local _, frame = cap:read{}

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones() -- reconstruct clones inside the language model
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------

local function run()
  protos.cnn:evaluate()
  protos.lm:evaluate()

  while true do
    local w = frame:size(2)
    local h = frame:size(1)

    -- take a central crop
    local crop = cv.getRectSubPix{image=frame, patchSize={h,h}, center={w/2, h/2}}
    local cropsc = cv.resize{src=crop, dsize={256,256}}
    -- BGR2RGB
    cropsc = cropsc:index(3,torch.LongTensor{3,2,1})
    -- HWC2CHW
    cropsc = cropsc:permute(3,1,2)

    -- fetch a batch of data
    local batch = cropsc:contiguous():view(1,3,256,256)
    local batch_processed = net_utils.prepro(batch, false, opt.gpuid >= 0) -- preprocess in place, and don't augment

    -- forward the model to get loss
    local feats = protos.cnn:forward(batch_processed)

    -- forward the model to also get generated samples for each image
    local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
    local seq = protos.lm:sample(feats, sample_opts)
    local sents = net_utils.decode_sequence(vocab, seq)

    print(sents[1])

    cv.putText{
      img=crop,
      text = sents[1],
      org={10,20},
      fontFace=cv.FONT_HERSHEY_DUPLEX,
      fontScale=0.5,
      color={255, 255, 0},
      thickness=1
    }

    cv.imshow{winname="NeuralTalk2", image=crop}
    if cv.waitKey{30} >= 0 then break end

    cap:read{image=frame}
  end
end

run()
