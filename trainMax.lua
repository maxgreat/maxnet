require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'

function computeScore(output, target)

  -- Coputes the top1 and top5 error rate
  local batchSize = output:size(1)
  print('Output:')
  print(output:size())
  local _ , predictions = output:float():sort(2, true) -- descending

  -- Find which predictions match the target
  local correct = predictions:eq(
  target:long():view(batchSize, 1):expandAs(output))

  -- Top-1 score
  local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

  -- Top-5 score, if there are at least 5 classes
  local len = math.min(5, correct:size(2))
  local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

  return top1 * 100, top5 * 100
end

-- Main --
local cmd = torch.CmdLine()

cmd:text()
cmd:text('Train a model with the output of another already trained model')
cmd:text()
cmd:text('Options:')
------------ General options --------------------
cmd:option('-trainList',       'trainList.txt',         'Path to the training images file')
cmd:option('-output',    'gen', 'output to save model')
cmd:option('-trainedModel', 'models/resnet-152.t7', 't7 model file')
cmd:option('-batchSize', '128', 'Batch size')
cmd:text()

local opt = cmd:parse(arg or {})

batch_size = tonumber(opt.batchSize)

-- Load the model to train
local model = require 'models/smallmaxnet' --load the model
model = model({}) --load the model
print('Model to train :')
print(model)

local modelTrained = torch.load(opt.trainedModel)
-- Remove the fully connected layer
assert(torch.type(modelTrained:get(#modelTrained.modules)) == 'nn.Linear')
modelTrained:remove(#modelTrained.modules)
modelTrained = modelTrained:cuda()

-- crit = nn.CrossEntropyCriterion():cuda() --cross  entropy criterium load on cuda
crit = nn.SmoothL1Criterion():cuda()

image_list = {}
label_list = {}

--read train file
print("Creating image and label list from file")
input_file_train = "datasets/trainList.txt"
i = 0
for line in io.lines(opt.trainList) do
	--split the line in image path and label
	local image_data = {}
	for word in line:gmatch("%S+") do
		table.insert(image_data, word)
	end
	
	table.insert(image_list, image_data[1])	
end
print("Nb image for training "..#image_list)

local top1Sum, top5Sum = 0.0, 0.0

nbepoch = 1 --number of epoch to do
for n = 1, nbepoch do
	print("Epoch "..n)
	--shuffle at each epoch
	shuffle = torch.randperm(#image_list)    
	
	-- loop to handle every images
	print("Nb images : "..#image_list)
	print("batch size : "..batch_size)
	iter = (#image_list) / batch_size
	print("Iteration :"..iter)
	N = 0
	for i=1,iter do
		local timer = torch.Timer()
		input = torch.Tensor(batch_size, 3,256,256)
		target = torch.Tensor(batch_size, 2048)
		for j=1,batch_size do
			el = shuffle[j + (i-1)*batch_size]
			input[j] = image.load(image_list[shuffle[el]],3, 'float')
		end
		--target = target:cuda()
		input = input:cuda()
		
		--compute the target
		-- resnet need 224x224 images ! TODO
		target = modelTrained:forward(input)
		print("Target size:")
		print(target:size())

		--print(input:size())
		output = model:forward(input)
		loss = crit:forward(output, target)
		
		print("Loss : ")
		print(loss)
		
		crit:backward(output, target)
		model:backward(input, crit.gradInput)

		-- Compute score
		local top1, top5 = computeScore(output, target, 0)
		top1Sum = top1Sum + top1*batch_size
	    	top5Sum = top5Sum + top5*batch_size
	    	N = N + batch_size

	    	print((' | Test: [%d][%d/%d]    Time %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
	      	n, i, iter, timer:time().real, top1, top1Sum / N, top5, top5Sum / N))
	end
	torch.save(output..'/smallmax-'..n..'.t7', model)
	
end





