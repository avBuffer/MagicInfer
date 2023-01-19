#ifndef MAGIC_DATA_LOAD_DATA_HPP_
#define MAGIC_DATA_LOAD_DATA_HPP_

#include <armadillo>
#include <string>

using namespace std;


namespace magic_infer 
{

class CSVDataLoader 
{
public:
    /**
     * 从csv文件中初始化张量
     * @param file_path csv文件的路径
     * @param split_char 分隔符号
     * @return 根据csv文件得到的张量
     */
    static arma::fmat LoadData(const string &file_path, char split_char = ',');

private:
    /**
     * 得到csv文件的尺寸大小
     * @param file csv文件的路径
     * @param split_char 分割符号
     * @return 根据csv文件的尺寸大小
     */
    static pair<size_t, size_t> GetMatrixSize(ifstream &file, char split_char);
};

}
#endif //MAGIC_DATA_LOAD_DATA_HPP_
