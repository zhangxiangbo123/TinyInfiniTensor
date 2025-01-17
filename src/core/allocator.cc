#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        this->used += size;
        for (auto it = this->freeBlocks.begin(); it != this->freeBlocks.end(); it++){
            if (it->second >= size){
                size_t addr = it -> first;
                if (it ->second > size){
                    this->freeBlocks[addr + size] = it->second - size;
                }
                this->freeBlocks.erase(it);
                return addr;
            }
        }

        size_t newAddr = this->peak;
        this->peak += size;
        return newAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        this->used -= size;
        if (addr + size == this->peak) {
            this->peak -= size;
            this->freeBlocks[addr] = size;
            return;
        }   

        for (auto it = this->freeBlocks.begin(); it != this->freeBlocks.end(); it ++){
            // 向前合并
            if (it->first + it->second == addr){
                it->second +=size;
                auto tmp = this->freeBlocks.find(addr);
                this->freeBlocks.erase(tmp);
                return;
            }

            // 向后合并
            if (addr + size == it->first){
                this->freeBlocks[addr] = size + it->second;
                this->freeBlocks.erase(it);
                return;
            }
        }

        // 无法合并
        this->freeBlocks[addr] = size;
        return;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
