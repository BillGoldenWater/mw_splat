---@type ClientUIContainerControl
local this = script.object

local controlId = 1
local imageId = 106057

function OnStart()
	local splats = require("splats")
	local bg = game.InstantiateClientUIControl(controlId, this)
	---@cast bg ClientUIImageControl
	bg.sizeDeltaY = 900
	bg.sizeDeltaX = bg.sizeDeltaY * splats.aspect_ratio

	bg:SetImage(Enum.ImageSource.StaticReference, 100001)
	bg.imageColor = Color(splats.background[1], splats.background[2], splats.background[3], splats.background[4])
	bg.imageType = Enum.ImageType.Stretch
	bg.enableMask = false
	bg.fillType = Enum.ImageFillType.Unused

	for _, v in ipairs(splats.splats) do
		local img = game.InstantiateClientUIControl(controlId, this)
		---@cast img ClientUIImageControl

		img.anchoredPositionX = v[1]
		img.anchoredPositionY = v[2]
		img.sizeDeltaX = v[3]
		img.sizeDeltaY = v[4]
		img.localRotationZ = v[5]
		img.localScaleX = v[10]
		img.localScaleY = v[11]

		img:SetImage(Enum.ImageSource.StaticReference, imageId)
		img.imageColor = Color(v[6], v[7], v[8], v[9])
		img.imageType = Enum.ImageType.Stretch
		img.enableMask = false
		img.fillType = Enum.ImageFillType.Unused
	end
end
