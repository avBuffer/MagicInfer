#ifndef PNNX_STOREZIP_H
#define PNNX_STOREZIP_H

#include <map>
#include <string>
#include <vector>

using namespace std;


namespace pnnx 
{

class StoreZipReader
{
public:
    StoreZipReader();
    ~StoreZipReader();

    int open(const string& path);
    size_t get_file_size(const string& name);
    int read_file(const string& name, char* data);
    int close();

private:
    FILE* fp;

    struct StoreZipMeta
    {
        size_t offset;
        size_t size;
    };

    map<string, StoreZipMeta> filemetas;
};


class StoreZipWriter
{
public:
    StoreZipWriter();
    ~StoreZipWriter();

    int open(const string& path);
    int write_file(const string& name, const char* data, size_t size);
    int close();

private:
    FILE* fp;

    struct StoreZipMeta
    {
        string name;
        size_t lfh_offset;
        uint32_t crc32;
        uint32_t size;
    };

    vector<StoreZipMeta> filemetas;
};

} // namespace pnnx
#endif // PNNX_STOREZIP_H
