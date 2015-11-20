--[[
Test the LanguageModel implementation
--]]

require 'torch'
require 'LanguageModel'

local gradcheck = require 'misc.gradcheck'

local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given 
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end

-- Test the API of the Language Model
local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create LanguageModel instance
    local opt = {}
    opt.vocab_size = 5
    opt.input_encoding_size = 11
    opt.rnn_size = 8
    opt.num_layers = 2
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10
    local lm = nn.LanguageModel(opt)
    local crit = nn.LanguageModelCriterion()
    lm:type(dtype)
    crit:type(dtype)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0
    local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
    local output = lm:forward{imgs, seq}
    tester:assertlt(torch.max(output:view(-1)), 0) -- log probs should be <0

    -- the output should be of size (seq_length + 2, batch_size, vocab_size + 1)
    -- where the +1 is for the special END token appended at the end.
    tester:assertTensorSizeEq(output, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})

    local loss = crit:forward(output, seq)

    local gradOutput = crit:backward(output, seq)
    tester:assertTensorSizeEq(gradOutput, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})

    -- make sure the pattern of zero gradients is as expected
    local gradAbs = torch.max(torch.abs(gradOutput), 3):view(opt.seq_length+2, opt.batch_size)
    local gradZeroMask = torch.eq(gradAbs,0)
    local expectedGradZeroMask = torch.ByteTensor(opt.seq_length+2,opt.batch_size):zero()
    expectedGradZeroMask[{ {1}, {} }]:fill(1) -- first time step should be zero grad (img was passed in)
    expectedGradZeroMask[{ {6,9}, 1 }]:fill(1)
    expectedGradZeroMask[{ {7,9}, 6 }]:fill(1)
    tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

    local gradInput = lm:backward({imgs, seq}, gradOutput)
    tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.input_encoding_size})
    tester:asserteq(gradInput[2]:nElement(), 0, 'grad on seq should be empty tensor')

    -- also double check that we can use scheduled sampling without errors
    lm:setScheduledSamplingEpsilon(0) -- 0 ensures we definitely sample
    local output = lm:forward{imgs, seq}

  end
  return f
end

-- test just the language model alone (without the criterion)
local function gradCheckLM()

  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local output = lm:forward{imgs, seq}
  local w = torch.randn(output:size(1), output:size(2), output:size(3))
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  local gradOutput = w
  local gradInput, dummy = unpack(lm:backward({imgs, seq}, gradOutput))

  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward{x, seq}
    local loss = torch.sum(torch.cmul(output, w))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end

local function gradCheck()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local output = lm:forward{imgs, seq}
  local loss = crit:forward(output, seq)
  local gradOutput = crit:backward(output, seq)
  local gradInput, dummy = unpack(lm:backward({imgs, seq}, gradOutput))

  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward{x, seq}
    local loss = crit:forward(output, seq)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

local function overfit()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 7
  opt.rnn_size = 24
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local params, grad_params = lm:getParameters()
  print('number of parameters:', params:nElement(), grad_params:nElement())
  local lstm_params = 4*(opt.input_encoding_size + opt.rnn_size)*opt.rnn_size + opt.rnn_size*4*2
  local output_params = opt.rnn_size * (opt.vocab_size + 1) + opt.vocab_size+1
  local table_params = (opt.vocab_size + 1) * opt.input_encoding_size
  local expected_params = lstm_params + output_params + table_params
  print('expected:', expected_params)

  local function lossFun()
    grad_params:zero()
    local output = lm:forward{imgs, seq}
    local loss = crit:forward(output, seq)
    local gradOutput = crit:backward(output, seq)
    lm:backward({imgs, seq}, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  for t=1,30 do
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.vocab_size+1) - loss), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/30: loss %f', t, loss))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
end

-- check that we can call :sample() and that correct-looking things happen
local function sample()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModel(opt)

  local imgs = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
  local seq = lm:sample(imgs)

  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 1)
  tester:assertle(torch.max(seq), opt.vocab_size+1)
  print('\nsampled sequence:')
  print(seq)
end

tests.doubleApiForwardTest = forwardApiTestFactory('torch.DoubleTensor')
tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
tests.cudaApiForwardTest = forwardApiTestFactory('torch.CudaTensor')
tests.gradCheck = gradCheck
tests.gradCheckLM = gradCheckLM
tests.overfit = overfit
tests.sample = sample

tester:add(tests)
tester:run()
