local gradcheck = {}


function gradcheck.relative_error(x, y, h)
  h = h or 1e-12
  if torch.isTensor(x) and torch.isTensor(y) then
    local top = torch.abs(x - y)
    local bottom = torch.cmax(torch.abs(x) + torch.abs(y), h)
    return torch.max(torch.cdiv(top, bottom))
  else
    return math.abs(x - y) / math.max(math.abs(x) + math.abs(y), h)
  end
end


function gradcheck.numeric_gradient(f, x, df, eps)
  df = df or 1.0
  eps = eps or 1e-8
  local n = x:nElement()
  local x_flat = x:view(n)
  local dx_num = x.new(#x):zero()
  local dx_num_flat = dx_num:view(n)
  for i = 1, n do
    local orig = x_flat[i]
    
    x_flat[i] = orig + eps
    local pos = f(x)
    if torch.isTensor(df) then
      pos = pos:clone()
    end
    
    x_flat[i] = orig - eps
    local neg = f(x)
    if torch.isTensor(df) then
      neg = neg:clone()
    end
    
    local d = nil
    if torch.isTensor(df) then
      d = torch.dot(pos - neg, df) / (2 * eps)
    else
      d = df * (pos - neg) / (2 * eps)
    end
    
    dx_num_flat[i] = d
    x_flat[i] = orig
  end
  return dx_num
end


--[[
Inputs:
- f is a function that takes a tensor and returns a scalar
- x is the point at which to evalute f
- dx is the analytic gradient of f at x
--]]
function gradcheck.check_random_dims(f, x, dx, eps, num_iterations, verbose)
  if verbose == nil then verbose = false end
  eps = eps or 1e-4

  local x_flat = x:view(-1)
  local dx_flat = dx:view(-1)

  local relative_errors = torch.Tensor(num_iterations)

  for t = 1, num_iterations do
    -- Make sure the index is really random.
    -- We have to call this on the inner loop because some functions
    -- f may be stochastic, and eliminating their internal randomness for
    -- gradient checking by setting a manual seed. If this is the case,
    -- then we will always sample the same index unless we reseed on each
    -- iteration.
    torch.seed()
    local i = torch.random(x:nElement())

    local orig = x_flat[i]
    x_flat[i] = orig + eps
    local pos = f(x)

    x_flat[i] = orig - eps
    local neg = f(x)
    local d_numeric = (pos - neg) / (2 * eps)
    local d_analytic = dx_flat[i]

    x_flat[i] = orig

    local rel_error = gradcheck.relative_error(d_numeric, d_analytic)
    relative_errors[t] = rel_error
    if verbose then
      print(string.format('  Iteration %d / %d, error = %f',
            t, num_iterations, rel_error))
      print(string.format('  %f %f', d_numeric, d_analytic))
    end
  end
  return relative_errors
end


return gradcheck

  