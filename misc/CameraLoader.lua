--[[
Same as DataLoader but only requires a folder of images. 
Does not have an h5 dependency.
Only used at test time.
]]--

local utils = require 'misc.utils'
require 'lfs'
require 'image'
path = require 'pl.path'
require 'camera'

local CameraLoader = torch.class('CameraLoader')

function CameraLoader:__init(opt)
  local coco_json = utils.getopt(opt, 'coco_json', '')

  -- load the json file which contains additional information about the dataset
  print("Initialising CameraLoader")
    
  self.files = {}
  self.ids = {}

  self.N = #self.files
  print('CameraLoader found ' .. self.N .. ' images')

  self.iterator = 1
  self.cam = image.Camera(0) --initialise camera
  
end

function CameraLoader:resetIterator()
  self.iterator = 1
end

--[[
  Returns a batch of data:
  - X (N,3,256,256) containing the images as uint8 ByteTensor
  - info table of length N, containing additional information
  The data is iterated linearly in order
--]]
function CameraLoader:getBatch(opt)
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  -- pick an index of the datapoint to load next
  local img = self.cam:forward()
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256,256)
  local max_index = self.N
  local wrapped = false
  local infos = {}
  for i=1,batch_size do
    local ri = self.iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterator = ri_next

    -- load the image
    local img = self.cam:forward()
    image.save("tmp.jpg",img) -- for copying into vis/imgs
    img_batch_raw[i] = (image.scale(img, 256, 256)*(255/torch.max(img))) --to eliminate values above 255 and brighten image if too dark

    -- and record associated info as well
    local info_struct = {}
    info_struct.id = ri
    info_struct.file_path = "tmp.jpg"
    table.insert(infos, info_struct)
  end

  local data = {}
  data.images = img_batch_raw
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  data.infos = infos
  return data
end

