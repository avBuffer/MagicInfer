#ifndef MAGIC_LAYER_ABSTRACT_LAYER_FACTORY_HPP_
#define MAGIC_LAYER_ABSTRACT_LAYER_FACTORY_HPP_

#include <map>
#include <string>
#include <memory>

#include "layer.hpp"
#include "runtime/runtime_op.hpp"


namespace magic_infer 
{

class LayerRegisterer 
{
public:
    typedef ParseParameterAttrStatus (*Creator)(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &layer);
    typedef map<string, Creator> CreateRegistry;

    static void RegisterCreator(const string &layer_type, const Creator &creator);
    static shared_ptr<Layer> CreateLayer(const shared_ptr<RuntimeOperator> &op);
    static CreateRegistry &Registry();
};


class LayerRegistererWrapper 
{
public:
    LayerRegistererWrapper(const string &layer_type, const LayerRegisterer::Creator &creator) 
    {
        LayerRegisterer::RegisterCreator(layer_type, creator);
    }
};

}
#endif //MAGIC_LAYER_ABSTRACT_LAYER_FACTORY_HPP_
