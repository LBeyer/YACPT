#pragma once

#include <cuda.h>
#include <new>
#include <utility>
#include <type_traits>

template<class T>
class CudaPtr
{
public:
	CudaPtr();
	explicit CudaPtr(T* ptr);
	CudaPtr(const CudaPtr& other) = delete;
	CudaPtr(CudaPtr&& other);
	CudaPtr& operator=(const CudaPtr& other) = delete;
	CudaPtr& operator=(CudaPtr&& other);
	~CudaPtr();
	
	T* get();
	void reset(T* ptr);

private:
	void destroy();

	T* ptr;
};

template<class T>
class CudaPtr<T[]>
{
public:
	CudaPtr();
	explicit CudaPtr(T* ptr);
	CudaPtr(const CudaPtr& other) = delete;
	CudaPtr(CudaPtr&& other);
	CudaPtr& operator=(const CudaPtr& other) = delete;
	CudaPtr& operator=(CudaPtr&& other);
	~CudaPtr();

	T* get();
	void reset(T* ptr);

private:
	void destroy();

	T* ptr;
};

template<class T, class ...Args>
typename std::enable_if<!std::is_array<T>::value, CudaPtr<T>>::type
makeCudaPtr(Args&&... args)
{
	T* ptr = nullptr;
	cudaMalloc(&ptr, sizeof(T));
	assert(ptr != nullptr);
	T t(std::forward<Args>(args)...);
	cudaMemcpy(ptr, &t, sizeof(T), cudaMemcpyHostToDevice);
	return CudaPtr<T>(ptr);
}

template<class T, class ...Args>
typename std::enable_if<std::is_array<T>::value, CudaPtr<T>>::type
makeCudaPtr(size_t count, Args&&... args)
{
	typename std::decay<T>::type ptr = nullptr;
	cudaMalloc(&ptr, count * sizeof(std::remove_extent<T>::type));
	assert(ptr != nullptr);
	typename std::remove_extent<T>::type t(std::forward<Args>(args)...);
	for(size_t i = 0; i < count; i++)
	{
		cudaMemcpy(ptr + i, &t, sizeof(std::remove_extent<T>::type), cudaMemcpyHostToDevice);
	}
	return CudaPtr<T>(ptr);
}

template<class T>
CudaPtr<T>::CudaPtr()
	: ptr(nullptr)
{
}

template<class T>
CudaPtr<T>::CudaPtr(T* ptr)
	: ptr(ptr)
{
}

template<class T>
CudaPtr<T>::CudaPtr(CudaPtr&& other)
	: ptr(other.ptr)
{
	other.ptr = nullptr;
}

template<class T>
CudaPtr<T>& CudaPtr<T>::operator=(CudaPtr&& other)
{
	destroy();
	ptr = other.ptr;
	other.ptr = nullptr;
	return *this;
}

template<class T>
CudaPtr<T>::~CudaPtr()
{
	destroy();
}

template<class T>
T* CudaPtr<T>::get()
{
	return ptr;
}

template<class T>
void CudaPtr<T>::reset(T* ptr)
{
	destroy();
	this->ptr = ptr;
}

template<class T>
void CudaPtr<T>::destroy()
{
	if(ptr)
	{
		cudaFree(ptr);
	}
}

template<class T>
CudaPtr<T[]>::CudaPtr()
	: ptr(nullptr)
{
}

template<class T>
CudaPtr<T[]>::CudaPtr(T* ptr)
	: ptr(ptr)
{
}

template<class T>
CudaPtr<T[]>::CudaPtr(CudaPtr&& other)
	: ptr(other.ptr)
{
	other.ptr = nullptr;
}

template<class T>
CudaPtr<T[]>& CudaPtr<T[]>::operator=(CudaPtr&& other)
{
	destroy();
	ptr = other.ptr;
	other.ptr = nullptr;
	return *this;
}

template<class T>
CudaPtr<T[]>::~CudaPtr()
{
	destroy();
}

template<class T>
T* CudaPtr<T[]>::get()
{
	return ptr;
}

template<class T>
void CudaPtr<T[]>::reset(T* ptr)
{
	destroy();
	this->ptr = ptr;
}

template<class T>
void CudaPtr<T[]>::destroy()
{
	if(ptr)
	{
		cudaFree(ptr);
	}
}