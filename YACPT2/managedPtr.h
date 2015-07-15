#pragma once

#include "util.h"
#include <cuda.h>
#include <new>
#include <utility>
#include <type_traits>

template<class T>
class ManagedPtr
{
public:
	ManagedPtr();
	explicit ManagedPtr(T* ptr);
	ManagedPtr(const ManagedPtr&) = delete;
	ManagedPtr(ManagedPtr&& other);
	ManagedPtr& operator=(const ManagedPtr&) = delete;
	ManagedPtr& operator=(ManagedPtr&& other);
	~ManagedPtr();

	HOST DEVICE T* get() const;
	void reset(T* ptr);
	HOST DEVICE T* release();
	HOST DEVICE T* operator->() const;

private:
	void destroy();

	T* ptr;
};

template<class T>
class ManagedPtr<T[]>
{
public:
	ManagedPtr();
	ManagedPtr(T* ptr, size_t count);
	ManagedPtr(const ManagedPtr&) = delete;
	ManagedPtr(ManagedPtr&& other);
	ManagedPtr& operator=(const ManagedPtr&) = delete;
	ManagedPtr& operator=(ManagedPtr&& other);
	~ManagedPtr();

	HOST DEVICE T* get() const;
	void reset(T* ptr, size_t count);
	HOST DEVICE T* release();
	HOST DEVICE T* operator->() const;
	HOST DEVICE T& operator[](size_t index);
	HOST DEVICE const T& operator[](size_t index) const;
	HOST DEVICE T* begin();
	HOST DEVICE const T* begin() const;
	HOST DEVICE T* end();
	HOST DEVICE const T* end() const;

private:
	void destroy();

	size_t count;
	T* ptr;
};

template<class T, class ...Args>
typename std::enable_if<!std::is_array<T>::value, ManagedPtr<T>>::type
makeManaged(Args&&... args)
{
	T* ptr = nullptr;
	cudaMallocManaged(&ptr, sizeof(T));
	assert(ptr != nullptr);
	new(ptr)T(std::forward<Args>(args)...);
	return ManagedPtr<T>(ptr);
}

template<class T, class ...Args>
typename std::enable_if<std::is_array<T>::value, ManagedPtr<T>>::type
makeManaged(size_t count, Args&&... args)
{
	typename std::decay<T>::type ptr = nullptr;
	cudaMallocManaged(&ptr, count * sizeof(std::remove_extent<T>::type));
	assert(ptr != nullptr);
	for(size_t i = 0; i < count; i++)
	{
		new(ptr + i)typename std::remove_extent<T>::type(std::forward<Args>(args)...);
	}
	return{ptr, count};
}

template<class T>
ManagedPtr<T>::ManagedPtr()
	: ptr(nullptr)
{
}

template<class T>
ManagedPtr<T>::ManagedPtr(T* ptr)
	: ptr(ptr)
{
#ifdef _DEBUG
	cudaPointerAttributes attributes = {};
	assert(!ptr || cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess);
	assert(!ptr || attributes.isManaged == 1);
#endif
}

template<class T>
ManagedPtr<T>::ManagedPtr(ManagedPtr&& other)
	: ptr(other.release())
{
}

template<class T>
ManagedPtr<T>& ManagedPtr<T>::operator=(ManagedPtr&& other)
{
	reset(other.release());
	return *this;
}

template<class T>
ManagedPtr<T>::~ManagedPtr()
{
	destroy();
}

template<class T>
HOST DEVICE T* ManagedPtr<T>::get() const
{
	return ptr;
}

template<class T>
void ManagedPtr<T>::reset(T* ptr)
{
#ifdef _DEBUG
	cudaPointerAttributes attributes = {};
	assert(!ptr || cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess);
	assert(!ptr || attributes.isManaged == 1);
#endif

	destroy();
	this->ptr = ptr;
}

template<class T>
HOST DEVICE T* ManagedPtr<T>::release()
{
	auto ret = ptr;
	ptr = nullptr;
	return ret;
}

template<class T>
HOST DEVICE T* ManagedPtr<T>::operator->() const
{
	return ptr;
}

template<class T>
void ManagedPtr<T>::destroy()
{
	cudaDeviceSynchronize();
	if(ptr)
	{
		ptr->~T();
		cudaFree(ptr);
	}
}

template<class T>
ManagedPtr<T[]>::ManagedPtr()
	: ptr(nullptr),
	count(0)
{
}

template<class T>
ManagedPtr<T[]>::ManagedPtr(T* ptr, size_t count)
	: ptr(ptr),
	count(count)
{
#ifdef _DEBUG
	cudaPointerAttributes attributes = {};
	assert(!ptr || count > 0);
	assert(!ptr || cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess);
	assert(!ptr || attributes.isManaged == 1);
#endif
}

template<class T>
ManagedPtr<T[]>::ManagedPtr(ManagedPtr&& other)
	: ptr(other.release()),
	count(other.count)
{
}

template<class T>
ManagedPtr<T[]>::~ManagedPtr()
{
	destroy();
}

template<class T>
ManagedPtr<T[]>& ManagedPtr<T[]>::operator=(ManagedPtr&& other)
{
	reset(other.release(), other.count);
	return *this;
}

template<class T>
HOST DEVICE T* ManagedPtr<T[]>::get() const
{
	return ptr;
}

template<class T>
void ManagedPtr<T[]>::reset(T* ptr, size_t count)
{
#ifdef _DEBUG
	cudaPointerAttributes attributes = {};
	assert(!ptr || count > 0);
	assert(!ptr || cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess);
	assert(!ptr || attributes.isManaged == 1);
#endif

	destroy();
	this->ptr = ptr;
	this->count = count;
}

template<class T>
HOST DEVICE T* ManagedPtr<T[]>::release()
{
	auto ret = ptr;
	ptr = nullptr;
	return ret;
}

template<class T>
HOST DEVICE T* ManagedPtr<T[]>::operator->() const
{
	return ptr;
}

template<class T>
HOST DEVICE T& ManagedPtr<T[]>::operator[](size_t index)
{
	assert(index < count);
	return ptr[index];
}

template<class T>
HOST DEVICE const T& ManagedPtr<T[]>::operator[](size_t index) const
{
	assert(index < count);
	return ptr[index];
}

template<class T>
HOST DEVICE T* ManagedPtr<T[]>::begin()
{
	return ptr;
}

template<class T>
HOST DEVICE const T* ManagedPtr<T[]>::begin() const
{
	return ptr;
}

template<class T>
HOST DEVICE T* ManagedPtr<T[]>::end()
{
	return ptr + count;
}

template<class T>
HOST DEVICE const T* ManagedPtr<T[]>::end() const
{
	return ptr + count;
}

template<class T>
void ManagedPtr<T[]>::destroy()
{
	cudaDeviceSynchronize();
	if(ptr)
	{
		for(auto i = 0; i < count; i++)
		{
			(ptr + i)->~T();
		}
		cudaFree(ptr);
	}
}