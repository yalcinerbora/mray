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
	"Cameras": [{"id": 0, "type": "Pinhole"}],

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
	"Lights": [{ "id" : 0, "type" : "Null"}],
	"Mediums": [{ "id" : 0, "type" : "Vacuum"}],
	"Transforms":[{"id": 0, "type": "Identity"}],
	"Materials":[{"id": 0, "type": "Barycentric", "albedo" : [0, 1, 0]}],
	"Primitives":
	[{
		"id": 0,
		"type": "Triangle",
		"name": ".nodeTriangle",
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
		"transform": 0, "accelerator": 0,
		"material": 0,
		"primitive": 0
	}],
	"LightSurfaces": [],
	"CameraSurfaces": [{"camera": 0}]
}
)"};