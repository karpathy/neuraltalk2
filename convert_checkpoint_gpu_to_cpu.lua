--[[
A quick script for converting GPU checkpoints to CPU checkpoints.
CPU checkpoints are not saved by the training script automatically
because of Torch cloning limitations. In particular, it is not
possible to clone a GPU model on CPU, something like :clone():float()
with a single call, without needing extra memory on the GPU. If this
existed then it would be possible to do this inside the training
script without worrying about blowing up the memory.
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn' -- only needed if the loaded model used cudnn as backend. otherwise can be commented out
-- local imports
require 'misc.LanguageModel'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a GPU checkpoint to CPU checkpoint.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','GPU model checkpoint to convert')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.model)
local protos = checkpoint.protos

-- convert the networks to be CPU models
for k,v in pairs(protos) do
  print('converting ' .. k .. ' to CPU')
  if k == 'cnn' then
    protos[k] = cudnn.convert(v, nn):float()
  elseif k == 'lm' then
    v.clones = nil -- sanitize the clones inside the language model (if present just in case. but they shouldnt be)
    v.lookup_tables = nil
    protos[k]:float() -- ship to CPU
  else
    error('error: strange module in protos: ' .. k)
  end
end

local savefile = opt.model .. '_cpu.t7' -- append "cpu.t7" to filename
torch.save(savefile, checkpoint)
print('saved ' .. savefile)

