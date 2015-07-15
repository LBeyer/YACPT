#pragma once

#include <gl\glew.h>
#include <string>
#include <memory>
#include "cuda_gl_interop.h"
#include "cuda_runtime.h"
#include "resolution.h"
#include "util.h"

class Texture
{
public:
	explicit inline Texture(const Resolution& resolution);
	inline ~Texture();
	inline uchar4* map();
	inline void unmap();
	inline void save(const std::string& filename);

private:
	Resolution resolution;
	GLuint bufferObj;
	cudaGraphicsResource* resource;
	uchar4* devPtr;
};

inline Texture::Texture(const Resolution& resolution)
	: resolution(resolution),
	bufferObj(0),
	resource(nullptr),
	devPtr(nullptr)
{
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, resolution.width * resolution.height * 4, nullptr, GL_DYNAMIC_DRAW_ARB);
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}

inline Texture::~Texture()
{
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glDeleteBuffers(1, &bufferObj);
}

inline uchar4* Texture::map()
{
	cudaGraphicsMapResources(1, &resource);
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
	return devPtr;
}

inline void Texture::unmap()
{
	cudaGraphicsUnmapResources(1, &resource);
}

inline void Texture::save(const std::string& filename)
{
	std::unique_ptr<uchar4[]> data(new uchar4[resolution.width * resolution.height]);
	cudaMemcpy(data.get(), map(), resolution.width * resolution.height * sizeof(uchar4), cudaMemcpyDeviceToHost);
	unmap();

	const auto scanlinebytes = resolution.width * 3;
	const auto padding = (4 - (scanlinebytes % 4)) % 4;
	const auto paddedScanlineWidth = scanlinebytes + padding;
	std::unique_ptr<unsigned char[]> bgrBuffer(new unsigned char[resolution.height * paddedScanlineWidth]);
	for(auto x = 0u; x < resolution.width; x++)
	{
		for(auto y = 0u; y < resolution.height; y++)
		{
			auto dataBufferPos = y * resolution.width + x;
			auto bgrBufferPos = y * paddedScanlineWidth + (3 * x);
			bgrBuffer[bgrBufferPos] = data[dataBufferPos].z;
			bgrBuffer[bgrBufferPos + 1] = data[dataBufferPos].y;
			bgrBuffer[bgrBufferPos + 2] = data[dataBufferPos].x;
		}
	}

	const auto filesize = 54 + resolution.height * paddedScanlineWidth;

	const unsigned char header[54] = {
		'B', 'M', // magic
		static_cast<unsigned char>(filesize), static_cast<unsigned char>(filesize >> 8), static_cast<unsigned char>(filesize >> 16), static_cast<unsigned char>(filesize >> 24), // size
		0, 0, // app data
		0, 0, // app data
		54, 0, 0, 0, // data offset
		40, 0, 0, 0, // info header size
		static_cast<unsigned char>(resolution.width), static_cast<unsigned char>(resolution.width >> 8), static_cast<unsigned char>(resolution.width >> 16), static_cast<unsigned char>(resolution.width >> 24), // width
		static_cast<unsigned char>(resolution.height), static_cast<unsigned char>(resolution.height >> 8), static_cast<unsigned char>(resolution.height >> 16), static_cast<unsigned char>(resolution.height >> 24), // height
		1, 0, // number color planes
		24, 0, // bits per pixel
		0, 0, 0, 0, // compression
		0, 0, 0, 0, // image bits size
		0x13, 0x0B, 0, 0, // horizontal resolution in pixel / m
		0x13, 0x0B, 0, 0, // vertical resolution (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
		0, 0, 0, 0, // #colors in pallete
		0, 0, 0, 0, // #important colors
	};

	FILE* f = nullptr;
	fopen_s(&f, filename.c_str(), "wb");
	if(f)
	{
		fwrite(header, 1, 54, f);
		fwrite(bgrBuffer.get(), 1, resolution.height * paddedScanlineWidth, f);
		fclose(f);
	}
}