
#ifndef MAGIC_PARSER_PARSE_EXPRESSION_HPP_
#define MAGIC_PARSER_PARSE_EXPRESSION_HPP_

#include <string>
#include <utility>
#include <vector>
#include <memory>

using namespace std;


namespace magic_infer 
{

enum class TokenType 
{
    TokenUnknown      = -1,
    TokenInputNumber  = 0,
    TokenComma        = 1,
    
    TokenAdd          = 2,
    TokenMul          = 3,
    
    TokenLeftBracket  = 4,
    TokenRightBracket = 5,
};

struct Token 
{
    TokenType token_type = TokenType::TokenUnknown;
    int32_t start_pos = 0; //词语开始的位置
    int32_t end_pos = 0; // 词语结束的位置
    
    Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
        : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {}
};

struct TokenNode 
{
    int32_t num_index = -1;
    shared_ptr<TokenNode> left  = nullptr;
    shared_ptr<TokenNode> right = nullptr;

    TokenNode(int32_t num_index, shared_ptr<TokenNode> left, shared_ptr<TokenNode> right);
    TokenNode() = default;
};


// add(add(add(@0,@1),@1),add(@0,@2))
class ExpressionParser 
{
public:
    explicit ExpressionParser(string statement) : statement_(move(statement)) {}

    void Tokenizer(bool need_retoken = false);
    vector<shared_ptr<TokenNode>> Generate();

    const vector<Token> &tokens() const;
    const vector<string> &token_strs() const;

 private:
    shared_ptr<TokenNode> Generate_(int32_t &index);
    vector<Token> tokens_;
    vector<string> token_strs_;
    string statement_;
};

}
#endif //MAGIC_PARSER_PARSE_EXPRESSION_HPP_
