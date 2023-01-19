
#include "data/load_data.hpp"
#include <string>
#include <fstream>
#include <armadillo>
#include <utility>
#include <glog/logging.h>


namespace magic_infer 
{

arma::fmat CSVDataLoader::LoadData(const string &file_path, const char split_char) 
{
    CHECK(!file_path.empty()) << "File path is empty!";
    ifstream in(file_path);
    CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;

    arma::fmat data;
    string line_str;
    stringstream line_stream;

    const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
    data.zeros(rows, cols);

    size_t row = 0;
    while (in.good()) {
        getline(in, line_str);
        if (line_str.empty()) break;

        string token;
        line_stream.clear();
        line_stream.str(line_str);

        size_t col = 0;
        while (line_stream.good()) {
            getline(line_stream, token, split_char);
            try {
                data.at(row, col) = stof(token);
            } catch (exception &e) {
                LOG(ERROR) << "Parse CSV File meet error: " << e.what();
                continue;
            }
            
            col += 1;
            CHECK(col <= cols) << "There are excessive elements on the column";
        }

        row += 1;
        CHECK(row <= rows) << "There are excessive elements on the row";
    }
    return data;
}


pair<size_t, size_t> CSVDataLoader::GetMatrixSize(ifstream &file, char split_char) 
{
    bool load_ok = file.good();
    file.clear();
    size_t fn_rows = 0;
    size_t fn_cols = 0;
    const ifstream::pos_type start_pos = file.tellg();

    string token;
    string line_str;
    stringstream line_stream;

    while (file.good() && load_ok) {
        getline(file, line_str);
        if (line_str.empty()) break;

        line_stream.clear();
        line_stream.str(line_str);
        size_t line_cols = 0;

        string row_token;
        while (line_stream.good()) {
            getline(line_stream, row_token, split_char);
            ++line_cols;
        }

        if (line_cols > fn_cols) fn_cols = line_cols;
        ++fn_rows;
    }

    file.clear();
    file.seekg(start_pos);
    return {fn_rows, fn_cols};
}

}
