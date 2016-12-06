require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'


-- Main --
local cmd = torch.CmdLine()

cmd:text()
cmd:text('Train a model with the output of another already trained model')
cmd:text()
cmd:text('Options:')
------------ General options --------------------
cmd:option('-imageList',       'imageList.txt',         'Path to the training images file')
cmd:option('-output',    'features.txt', 'output to save features')
cmd:option('-model', 'resnet-152.t7', 't7 model file')
cmd:option('-batchSize', '128', 'Batch size')
cmd:option('-fc', 'no', 'Keep the fully connected layer')
cmd:text()

local opt = cmd:parse(arg or {})

-- Convert batch Size
batchSize = tonumber(opt.batchSize)

-- Timer
timer = torch.Timer()

-- Load the model
local model = torch.load(opt.model)

-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)
model = model:cuda()

image_list = {}
label_list = {}

--read image list file
print("Read image file : "..opt.imageList)
file = assert(io.open(opt.imageList, "r"))
size = file:seek("end") --size of the file
file:seek("set", 0) -- return to the beginning of the file

print('File zie :')
print(size)

while true do 
	line = file:read()
	if not line then break end
	io.write('\r')
	io.write(file:seek()/size)
	table.insert(image_list, line)
end
print(" ")

-- The model was trained with this input normalization
meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

--local transform = t.Compose{
--   t.Scale(256),
--   t.ColorNormalize(meanstd),
--   t.CenterCrop(224),
--}

function transformImage(inp)
	local im = image.scale(inp,256,256) --scale to 256
	-- normalize color
	for i=1,3 do
		im[i]:add(-meanstd.mean[i])
		im[i]:div(meanstd.std[i])
	end
	width = math.ceil( (im:size(3)-224) / 2)
	height = math.ceil( (im:size(2)-224) / 2)
	return image.crop(im, width, height, width+224, height+224)
end


file = io.open(opt.output, 'w')
print("Extract features")
for i=1,#image_list/batchSize do
	io.write('\r')
	io.write(i/ (#image_list/batchSize)) 
	input = torch.Tensor(batchSize, 3,224,224)
	for j=1,batchSize do 
		local im = image.load(image_list[i*j], 3, 'float') --load image
		input[j] = transformImage(im) --transform image, usefull?
	end
	features = model:forward(input:cuda())
	for k=1,batchSize do
		file:write(image_list[i]..' ')
		--print(features:size())
		for j=1,features:size()[2] do
		--for j=1,10 do
			file:write(features[k][j]..' ')
		end
		file:write('\n')	
	end
	
end





