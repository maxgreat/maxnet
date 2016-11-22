require "image"
require "nn"
require 'cunn'


lr = 0.01
nr_epochs = 20
image_path = "/home/mrim/budnik/clicide/clicide_128"
image_ext = "JPG"

input_file = "/home/mrim/budnik/scripts/siamese_net/train_128.txt"

files = {}

for file in paths.files(image_path) do
	if not file:find('wall') then 
		if file:find(image_ext .. '$') then
			table.insert(files,paths.concat(image_path,file))
		end
	end
end


images = {}
files_dic ={}

for i,file in ipairs(files) do
	table.insert(images, image.load(file))
	files_dic[file] = i
end


-- net definition:

siam = torch.load('/home/mrim/budnik/scripts/siamese_net/models/siam_net_24.t7')

siam:cuda()

crit = nn.HingeEmbeddingCriterion()

crit:cuda()


local iter = 0
local total_err = 0
local epoch_nr = 1
while epoch_nr <= nr_epochs do
	local train_pair_file = io.open(input_file,'r')
	for line in train_pair_file:lines() do
		local l = line:split(' ')
		target = tonumber(l[3])
		
        local input = torch.Tensor(2,3,128,128)
        --local input = torch.cat(images[files_dic[l[1]]],images[files_dic[l[2]]],1)
        --print(input:size())
		input[1] = images[files_dic[l[1]]]
		input[2] = images[files_dic[l[2]]]

		--input = {}
		--table.insert(input,images[files_dic[l[1]]])
		--table.insert(input,images[files_dic[l[2]]])

		input = input:float():cuda()
		--local input = {images[files_dic[l[1]]], images[files_dic[l[2]]]}
		--print (input)
		siam:zeroGradParameters() 

		--local out = siam:forward(input)
		local out = siam:forward(input)

		

		local err =  crit:forward(out, target)
		total_err = total_err + err
		gradOut = crit:backward(out,target)
		local gradInp = siam:backward(input, gradOut)
		siam:updateParameters(lr)
		if iter % 200 == 0 then
			print('Current step: ' .. iter .. ' epoch: ' .. epoch_nr .. ' and error: ' .. total_err)
			--print(out)
			total_err = 0
			print(siam.modules[1].output[1])
			--print(siam.modules[1].modules[1].modules[10].output)
			--print(siam.modules[1].modules[2].modules[10])

		end
		iter = iter + 1
	end
	--sprint ('Current epoch number: ' .. epoch_nr)
	train_pair_file:close()
	torch.save('/home/mrim/budnik/scripts/siamese_net/models/siam_net_' .. 24+epoch_nr .. '.t7', siam)
	epoch_nr = epoch_nr + 1
	collectgarbage()
end


print("Done.")


