#include "parser/parse_expression.hpp"

#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>
#include <glog/logging.h>


namespace magic_infer 
{

void ReversePolish(const shared_ptr<TokenNode> &root_node, vector<shared_ptr<TokenNode>> &reverse_polish) 
{
    if (root_node != nullptr) {
        ReversePolish(root_node->left, reverse_polish);
        ReversePolish(root_node->right, reverse_polish);
        reverse_polish.push_back(root_node);
    }
}


void ExpressionParser::Tokenizer(bool need_retoken) 
{
    if (!need_retoken && !this->tokens_.empty()) return;

    CHECK(!statement_.empty()) << "The input statement is empty!";
    statement_.erase(remove_if(statement_.begin(), statement_.end(), [](char c) { return isspace(c); }), statement_.end());
    CHECK(!statement_.empty()) << "The input statement is empty!";

    for (int32_t i = 0; i < statement_.size();) {
        char c = statement_.at(i);
        if (c == 'a') {
            CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd') << "Parse add token failed, illegal character: " << c;
            CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd') << "Parse add token failed, illegal character: " << c;
            Token token(TokenType::TokenAdd, i, i + 3);
            tokens_.push_back(token);
            string token_operation = string(statement_.begin() + i, statement_.begin() + i + 3);
            token_strs_.push_back(token_operation);
            i = i + 3;
        
        } else if (c == 'm') {
            CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u') << "Parse add token failed, illegal character: " << c;
            CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l') << "Parse add token failed, illegal character: " << c;
            Token token(TokenType::TokenMul, i, i + 3);
            tokens_.push_back(token);
            string token_operation = string(statement_.begin() + i, statement_.begin() + i + 3);
            token_strs_.push_back(token_operation);
            i = i + 3;
        
        } else if (c == '@') {
            CHECK(i + 1 < statement_.size() && isdigit(statement_.at(i + 1))) << "Parse number token failed, illegal character: " << c;
            int32_t j = i + 1;
            for (; j < statement_.size(); ++j) {
                if (!isdigit(statement_.at(j))) break;
            }

            Token token(TokenType::TokenInputNumber, i, j);
            CHECK(token.start_pos < token.end_pos);
            tokens_.push_back(token);
            string token_input_number = string(statement_.begin() + i, statement_.begin() + j);
            token_strs_.push_back(token_input_number);
            i = j;
        
        } else if (c == ',') {
            Token token(TokenType::TokenComma, i, i + 1);
            tokens_.push_back(token);
            string token_comma = string(statement_.begin() + i, statement_.begin() + i + 1);
            token_strs_.push_back(token_comma);
            i += 1;
        
        } else if (c == '(') {
            Token token(TokenType::TokenLeftBracket, i, i + 1);
            tokens_.push_back(token);
            string token_left_bracket = string(statement_.begin() + i, statement_.begin() + i + 1);
            token_strs_.push_back(token_left_bracket);
            i += 1;
        
        } else if (c == ')') {
            Token token(TokenType::TokenRightBracket, i, i + 1);
            tokens_.push_back(token);
            string token_right_bracket = string(statement_.begin() + i, statement_.begin() + i + 1);
            token_strs_.push_back(token_right_bracket);
            i += 1;
        
        } else {
            LOG(FATAL) << "Unknown    illegal character: " << c;
        }
    }
}


const vector<Token> &ExpressionParser::tokens() const 
{
    return this->tokens_;
}


const vector<string> &ExpressionParser::token_strs() const 
{
    return this->token_strs_;
}


shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t &index) 
{
    CHECK(index < this->tokens_.size());
    const auto current_token = this->tokens_.at(index);
    CHECK(current_token.token_type == TokenType::TokenInputNumber || 
          current_token.token_type == TokenType::TokenAdd || current_token.token_type == TokenType::TokenMul);
    
    if (current_token.token_type == TokenType::TokenInputNumber) {
        uint32_t start_pos = current_token.start_pos + 1;
        uint32_t end_pos = current_token.end_pos;
        CHECK(end_pos > start_pos);
        CHECK(end_pos <= this->statement_.length());
        const string &str_number = string(this->statement_.begin() + start_pos, this->statement_.begin() + end_pos);
        return make_shared<TokenNode>(stoi(str_number), nullptr, nullptr);

    } else if (current_token.token_type == TokenType::TokenMul || current_token.token_type == TokenType::TokenAdd) {
        shared_ptr<TokenNode> current_node = make_shared<TokenNode>();
        current_node->num_index = -int(current_token.token_type);

        index += 1;
        CHECK(index < this->tokens_.size());
        CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

        index += 1;
        CHECK(index < this->tokens_.size());
        const auto left_token = this->tokens_.at(index);

        if (left_token.token_type == TokenType::TokenInputNumber || 
            left_token.token_type == TokenType::TokenAdd || left_token.token_type == TokenType::TokenMul) {
            current_node->left = Generate_(index);
        } else {
            LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
        }

        index += 1;
        CHECK(index < this->tokens_.size());
        CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);

        index += 1;
        CHECK(index < this->tokens_.size());
        const auto right_token = this->tokens_.at(index);
        if (right_token.token_type == TokenType::TokenInputNumber || 
            right_token.token_type == TokenType::TokenAdd || right_token.token_type == TokenType::TokenMul) {
            current_node->right = Generate_(index);
        } else {
            LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
        }

        index += 1;
        CHECK(index < this->tokens_.size());
        CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
        return current_node;
    } else {
        LOG(FATAL) << "Unknown token type: " << int(current_token.token_type);
    }
}


vector<shared_ptr<TokenNode> > ExpressionParser::Generate() 
{
    if (this->tokens_.empty()) {
        this->Tokenizer(true);
    }

    int index = 0;
    shared_ptr<TokenNode> root = Generate_(index);
    CHECK(root != nullptr);
    CHECK(index == tokens_.size() - 1);

    // 转逆波兰式,之后转移到expression中
    vector<shared_ptr<TokenNode>> reverse_polish;
    ReversePolish(root, reverse_polish);

    for (const auto &node : reverse_polish) {
        LOG(INFO) << node->num_index;
    }
    return reverse_polish;
}


TokenNode::TokenNode(int32_t num_index, shared_ptr<TokenNode> left, shared_ptr<TokenNode> right) 
    : num_index(num_index), left(move(left)), right(move(right)) {}

}
