#ifndef PNNX_IR_H
#define PNNX_IR_H

#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;


#if BUILD_PNNX
namespace torch 
{
namespace jit 
{

struct Value;
struct Node;

} // namespace jit
} // namespace torch

namespace at 
{
class Tensor;
}
#endif // BUILD_PNNX


namespace pnnx 
{

class Parameter
{
public:
    Parameter() : type(0){}
    Parameter(bool _b) : type(1), b(_b) {}

    Parameter(int _i): type(2), i(_i) {} 
    Parameter(long _l) : type(2), i(_l) {}   
    Parameter(long long _l) : type(2), i(_l) {}
  
    Parameter(float _f) : type(3), f(_f) {}  
    Parameter(double _d) : type(3), f(_d) {}
  
    Parameter(const char* _s) : type(4), s(_s) {}
    Parameter(const string& _s) : type(4), s(_s) {}

    Parameter(const initializer_list<int>& _ai) : type(5), ai(_ai) {}
    Parameter(const initializer_list<int64_t>& _ai) : type(5) {
        for (const auto& x : _ai) ai.push_back((int)x);
    }
    Parameter(const vector<int>& _ai) : type(5), ai(_ai) {}

    Parameter(const initializer_list<float>& _af) : type(6), af(_af) {}
    Parameter(const initializer_list<double>& _af) : type(6) {
        for (const auto& x : _af) af.push_back((float)x);
    }
    Parameter(const vector<float>& _af) : type(6), af(_af) {}

    Parameter(const initializer_list<const char*>& _as) : type(7) {
        for (const auto& x : _as) as.push_back(string(x));
    }
    Parameter(const initializer_list<string>& _as) : type(7), as(_as) {}
    Parameter(const vector<string>& _as) : type(7), as(_as) {}

#if BUILD_PNNX
    Parameter(const torch::jit::Node* value_node);
    Parameter(const torch::jit::Value* value);
#endif // BUILD_PNNX

    static Parameter parse_from_string(const string& value);

    // 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
    int type;

    // value
    bool b;
    int i;
    float f;
    vector<int> ai;
    vector<float> af;

    // keep string typed member the last for cross cxxabi compatibility
    string s;
    vector<string> as;
};


bool operator==(const Parameter& lhs, const Parameter& rhs);

class Attribute
{
public:
    Attribute() : type(0) {}

#if BUILD_PNNX
    Attribute(const at::Tensor& t);
#endif // BUILD_PNNX

    Attribute(const initializer_list<int>& shape, const vector<float>& t);

    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
    int type;
    vector<int> shape;
    vector<char> data;
};


bool operator==(const Attribute& lhs, const Attribute& rhs);

// concat two attributes along the first axis
Attribute operator+(const Attribute& a, const Attribute& b);

class Operator;
class Operand
{
public:
    void remove_consumer(const Operator* c);

    Operator* producer;
    vector<Operator*> consumers;

    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=cp64 11=cp128 12=cp32
    int type;
    vector<int> shape;

    // keep string typed member the last for cross cxxabi compatibility
    string name;
    map<string, Parameter> params;

private:
    friend class Graph;
    Operand() {}
};


class Operator
{
public:
    vector<Operand*> inputs;
    vector<Operand*> outputs;

    // keep string typed member the last for cross cxxabi compatibility
    string type;
    string name;

    vector<string> inputnames;
    map<string, Parameter> params;
    map<string, Attribute> attrs;

private:
    friend class Graph;
    Operator() {}
};


class Graph
{
public:
    Graph();
    ~Graph();

    int load(const string& parampath, const string& binpath);
    int save(const string& parampath, const string& binpath);

    int python(const string& pypath, const string& binpath);
    int parse(const string& param);

    Operator* new_operator(const string& type, const string& name);
    Operator* new_operator_before(const string& type, const string& name, const Operator* cur);
    Operator* new_operator_after(const string& type, const string& name, const Operator* cur);

#if BUILD_PNNX
    Operand* new_operand(const torch::jit::Value* v);
#endif

    Operand* new_operand(const string& name);
    Operand* get_operand(const string& name);
    const Operand* get_operand(const string& name) const;

    vector<Operator*> ops;
    vector<Operand*> operands;

private:
    Graph(const Graph& rhs);
    Graph& operator=(const Graph& rhs);
};

} // namespace pnnx
#endif // PNNX_IR_H
