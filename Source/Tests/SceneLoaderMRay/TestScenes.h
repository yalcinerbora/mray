#pragma once

#include <string_view>

static constexpr std::string_view EmptyScene = std::string_view
{R"({})"};

static constexpr std::string_view MinimalValidScene = std::string_view
{R"(
{
	"Lights": [{ "id" : 0, "type" : "Null"}],
	"Mediums": [{ "id" : 0, "type" : "Vacuum"}],
	"Transforms":[{"id": 0, "type": "Identity"}],
	"Cameras":
	[{
		"id": 0, "type": "Pinhole",
		"fov": 0,
		"isFovX": true,
		"planes": [0,0],
		"gaze": [0,0,0],
		"position": [0,0,0],
		"up": [0,0,0]
	}],

	//
	"Boundary":
	{
		"medium": 0,
		"light" : 0,
		"transform": 0
	},
	"CameraSurfaces" : [{"camera": 0}],
	"LightSurfaces"  : [],
	"Surfaces"       : []
})"};

static constexpr std::string_view BasicScene = std::string_view
{R"(
{
	"Cameras":
	[{
		"id": 0, "type": "Pinhole",
		"fov": 0,
		"isFovX": true,
		"planes": [0,0],
		"gaze": [0,0,0],
		"position": [0,0,0],
		"up": [0,0,0]
	}],
	"Lights": [{ "id" : 0, "type" : "Null"}],
	"Mediums": [{ "id" : 0, "type" : "Vacuum"}],
	"Transforms":[{"id": 0, "type": "Identity"}],
	"Materials":[{"id": 0, "type": "Lambert", "albedo" : [0, 1, 0]}],
	"Primitives":
	[{
		"id": 0,
		"type": "Triangle",
		"tag": "nodeTriangle",
		"position": [[   0,  0.5, 0],
					 [-0.5, -0.5, 0],
					 [ 0.5, -0.5, 0]],
		"normal": [[0, 0, 1],
				   [0, 0, 1],
				   [0, 0, 1]],
		"uv": [[0.5, 1],
			   [  0, 0],
			   [  1, 0]]
	}],

	"Boundary":
	{
		"medium": 0,
		"light" : 0,
		"transform": 0
	},

	"Surfaces":
	[{
		"transform": 0, "material": 0, "primitive": 0
	}],
	"LightSurfaces": [],
	"CameraSurfaces": [{"camera": 0}]
}
)"};