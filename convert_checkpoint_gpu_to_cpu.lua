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
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed

local checkpoint = torch.load(opt.model)
local protos = checkpoint.protos

-------------------------------------------------------------------------------
-- these functions are adapted from Michael Partheil
-- https://groups.google.com/forum/#!topic/torch7/i8sJYlgQPeA
-- the problem is that you can't call :float() on cudnn module, it won't convert
function replaceModules(net, orig_class_name, replacer)
  local nodes, container_nodes = net:findModules(orig_class_name)
  for i = 1, #nodes do
    for j = 1, #(container_nodes[i].modules) do
      if container_nodes[i].modules[j] == nodes[i] then
        local orig_mod = container_nodes[i].modules[j]
        print('replacing a cudnn module with nn equivalent...')
        print(orig_mod)
        container_nodes[i].modules[j] = replacer(orig_mod)
      end
    end
  end
end
function cudnnNetToCpu(net)
  local net_cpu = net:clone():float()
  replaceModules(net_cpu, 'cudnn.SpatialConvolution', 
    function(orig_mod)
      local cpu_mod = nn.SpatialConvolution(orig_mod.nInputPlane, orig_mod.nOutputPlane,
          orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, orig_mod.padW, orig_mod.padH)
      cpu_mod.weight:copy(orig_mod.weight)
      cpu_mod.bias:copy(orig_mod.bias)
      cpu_mod.gradWeight = nil -- sanitize for thinner checkpoint
      cpu_mod.gradBias = nil -- sanitize for thinner checkpoint
      return cpu_mod
    end)
  replaceModules(net_cpu, 'cudnn.SpatialMaxPooling', 
    function(orig_mod)
      local cpu_mod = nn.SpatialMaxPooling(orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, 
                                           orig_mod.padW, orig_mod.padH)
      return cpu_mod
    end)
  replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
  return net_cpu
end
-------------------------------------------------------------------------------

-- convert the networks to be CPU models
for k,v in pairs(protos) do
  print('converting ' .. k .. ' to CPU')
  if k == 'cnn' then
    -- the cnn is a troublemaker
    local cpu_cnn = cudnnNetToCpu(v)
    protos[k] = cpu_cnn
  elseif k == 'lm' then
    local debugger = require('fb.debugger'); debugger:enter()
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

