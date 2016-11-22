local nn = require 'nn'
require 'cunn'
require 'cudnn'

local function createModel(opt)
  local model = nn.Sequential()

  model:add(cudnn.SpatialConvolution(3,32, 5,5, 1,1))
  model:add(cudnn.ReLU(true))
  model:add(nn.SpatialMaxPooling(3,3,2,2))
  
  model:add(cudnn.SpatialConvolution(32,64, 5,5, 1,1))
  model:add(cudnn.ReLU(true))
  model:add(nn.SpatialMaxPooling(3,3,3,3))

  model:add(cudnn.SpatialConvolution(64,128, 5,5, 1,1))
  model:add(cudnn.ReLU(true))
  model:add(nn.SpatialAveragePooling(3,3,3,3))

  model:add(cudnn.SpatialConvolution(128,128, 5,5, 1,1))
  model:add(cudnn.ReLU(true))
  model:add(nn.SpatialAveragePooling(2,2,2,2))

  model:add(nn.View(-1):setNumInputDims(3))

  
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
       local n = v.kW*v.kH*v.nOutputPlane
       v.weight:normal(0,math.sqrt(2/n))
       if cudnn.version >= 4000 then
          v.bias = nil
          v.gradBias = nil
       else
          v.bias:zero()
       end
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')

  model:cuda()

  if opt.cudnn == 'deterministic' then
    model:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end

  model:get(1).gradInput = nil

  return model
end

return createModel
