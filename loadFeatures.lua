require 'paths'
require 'image'

local features = {}

local function parseline(line)
	local j = 1
	feat = {}
	for i in string.gmatch(line, "%S+") do
  		if j == 1 then
			imageName = i
		else
			feat[j] = tonumber(i)
		end
		j = j + 1
	end		
	return imageName, feat
end


function features.loadFile(filename)
	-- Open file with verification
	local f = assert(io.open(filename, "r"))
	local feat = { imageList = {}, features = {}}
	-- Compute size
	size = f:seek("end") --size of the file
	f:seek("set", 0) --return to the beginning:w

	local i = 1	

	-- Read file
	while true do 
		local line = f:read()
		if not line then break end -- EOF
		io.write("\r"..f:seek()/size.." / "..size)
		feat.imageList[i], feat.features[i] = parseline(line)
		i = i + 1
	end
	return feat
end

return features

