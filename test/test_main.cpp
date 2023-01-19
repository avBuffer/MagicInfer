#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include <glog/logging.h>

using namespace std;


int main(int argc, char *argv[]) 
{
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("Magic");
    
    const string &out_path  = "../../log/";
    //判断该文件夹是否存在
    if (access(out_path.c_str(), 0) == -1) {
        if (mkdir(out_path.c_str(), S_IRWXU) != 0) { //创建失败
            cout << "Fail to create directory : " << out_path << endl;
            throw exception();
        }
    }
    
    FLAGS_log_dir = out_path;
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}
