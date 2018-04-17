#ifndef __PLUGIN_FACTORY_H__  
#define __PLUGIN_FACTORY_H__  
  
#include <algorithm>  
#include <cassert>  
#include <iostream>  
#include <cstring>  
#include <sys/stat.h>  
#include <map>  
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
#if 0
//Concat layer . TensorRT Concat only support cross channel
class ConcatPlugin : public IPlugin
{
public:
    ConcatPlugin(int axis){ _axis = axis; };
  //  ConcatPlugin(int axis, const void* buffer, size_t size);
	ConcatPlugin::ConcatPlugin(int axis, const void* buffer, size_t size)
	{
	    assert(size == (18*sizeof(int)));
	    const int* d = reinterpret_cast<const int*>(buffer);

	    dimsConv4_3 = DimsCHW{d[0], d[1], d[2]};
	    dimsFc7 = DimsCHW{d[3], d[4], d[5]};
	    dimsConv6 = DimsCHW{d[6], d[7], d[8]};
	    dimsConv7 = DimsCHW{d[9], d[10], d[11]};
	    dimsConv8 = DimsCHW{d[12], d[13], d[14]};
	    dimsConv9 = DimsCHW{d[15], d[16], d[17]};

	    _axis = axis;

	}


    inline int getNbOutputs() const override {return 1;};
  //  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override ;

	Dims  getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
	{
	    assert(nbInputDims == 6);

	    if(_axis == 1)
	    {
		top_concat_axis = inputs[0].d[0] + inputs[1].d[0] + inputs[2].d[0] + inputs[3].d[0] + inputs[4].d[0] + inputs[5].d[0];
		return DimsCHW(top_concat_axis, 1, 1);
	    }else if(_axis == 2){
		top_concat_axis = inputs[0].d[1] + inputs[1].d[1] + inputs[2].d[1] + inputs[3].d[1] + inputs[4].d[1] + inputs[5].d[1];
		return DimsCHW(2, top_concat_axis, 1);
	    }else{//_param.concat_axis == 3

		return DimsCHW(0, 0, 0);
	    }
	}


  //  int initialize() 


	int  initialize()  
	{
	    inputs_size = 6;//6个bottom层

	    if(_axis == 1)//c
	    {
		top_concat_axis = dimsConv4_3.c() + dimsFc7.c() + dimsConv6.c() + dimsConv7.c() + dimsConv8.c() + dimsConv9.c();
		bottom_concat_axis[0] = dimsConv4_3.c(); bottom_concat_axis[1] = dimsFc7.c(); bottom_concat_axis[2] = dimsConv6.c();
		bottom_concat_axis[3] = dimsConv7.c(); bottom_concat_axis[4] = dimsConv8.c(); bottom_concat_axis[5] = dimsConv9.c();

		concat_input_size_[0] = dimsConv4_3.h() * dimsConv4_3.w();  concat_input_size_[1] = dimsFc7.h() * dimsFc7.w();
		concat_input_size_[2] = dimsConv6.h() * dimsConv6.w();  concat_input_size_[3] = dimsConv7.h() * dimsConv7.w();
		concat_input_size_[4] = dimsConv8.h() * dimsConv8.w();  concat_input_size_[5] = dimsConv9.h() * dimsConv9.w();

		num_concats_[0] = dimsConv4_3.c(); num_concats_[1] = dimsFc7.c(); num_concats_[2] = dimsConv6.c();
		num_concats_[3] = dimsConv7.c(); num_concats_[4] = dimsConv8.c(); num_concats_[5] = dimsConv9.c();
	    }else if(_axis == 2){//h
		top_concat_axis = dimsConv4_3.h() + dimsFc7.h() + dimsConv6.h() + dimsConv7.h() + dimsConv8.h() + dimsConv9.h();
		bottom_concat_axis[0] = dimsConv4_3.h(); bottom_concat_axis[1] = dimsFc7.h(); bottom_concat_axis[2] = dimsConv6.h();
		bottom_concat_axis[3] = dimsConv7.h(); bottom_concat_axis[4] = dimsConv8.h(); bottom_concat_axis[5] = dimsConv9.h();

		concat_input_size_[0] = dimsConv4_3.w(); concat_input_size_[1] = dimsFc7.w(); concat_input_size_[2] = dimsConv6.w();
		concat_input_size_[3] = dimsConv7.w(); concat_input_size_[4] = dimsConv8.w(); concat_input_size_[5] = dimsConv9.w();

		num_concats_[0] = dimsConv4_3.c() * dimsConv4_3.h();  num_concats_[1] = dimsFc7.c() * dimsFc7.h();
		num_concats_[2] = dimsConv6.c() * dimsConv6.h();  num_concats_[3] = dimsConv7.c() * dimsConv7.h();
		num_concats_[4] = dimsConv8.c() * dimsConv8.h();  num_concats_[5] = dimsConv9.c() * dimsConv9.h();

	    }else{//_param.concat_axis == 3 , w
		top_concat_axis = dimsConv4_3.w() + dimsFc7.w() + dimsConv6.w() + dimsConv7.w() + dimsConv8.w() + dimsConv9.w();
		bottom_concat_axis[0] = dimsConv4_3.w(); bottom_concat_axis[1] = dimsFc7.w(); bottom_concat_axis[2] = dimsConv6.w();
		bottom_concat_axis[3] = dimsConv7.w(); bottom_concat_axis[4] = dimsConv8.w(); bottom_concat_axis[5] = dimsConv9.w();

		concat_input_size_[0] = 1; concat_input_size_[1] = 1; concat_input_size_[2] = 1;
		concat_input_size_[3] = 1; concat_input_size_[4] = 1; concat_input_size_[5] = 1;
		return 0;
	    }

	    return 0;
	}

  //  inline void terminate() override;
	void  terminate()
	{
	    //CUDA_CHECK(cudaFree(scale_data));
	    delete[] bottom_concat_axis;
	    delete[] concat_input_size_;
	    delete[] num_concats_;
	}



    inline size_t getWorkspaceSize(int) const override { return 0; };
    //int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
	{
	    float *top_data = reinterpret_cast<float*>(outputs[0]);
	    int offset_concat_axis = 0;
	    const bool kForward = true;
	    for (int i = 0; i < inputs_size; ++i) {
		const float *bottom_data = reinterpret_cast<const float*>(inputs[i]);

		const int nthreads = num_concats_[i] * concat_input_size_[i];
		//const int nthreads = bottom_concat_size * num_concats_[i];
		ConcatLayer(nthreads, bottom_data, kForward, num_concats_[i], concat_input_size_[i], top_concat_axis, bottom_concat_axis[i], offset_concat_axis, top_data, stream);

		offset_concat_axis += bottom_concat_axis[i];
	    }

	    return 0;
	}

   // size_t getSerializationSize() override;
	size_t getSerializationSize()
	{
	    return 18*sizeof(int);
	}


   // void serialize(void* buffer) override;

	void serialize(void* buffer)
	{
	    int* d = reinterpret_cast<int*>(buffer);
	    d[0] = dimsConv4_3.c(); d[1] = dimsConv4_3.h(); d[2] = dimsConv4_3.w();
	    d[3] = dimsFc7.c(); d[4] = dimsFc7.h(); d[5] = dimsFc7.w();
	    d[6] = dimsConv6.c(); d[7] = dimsConv6.h(); d[8] = dimsConv6.w();
	    d[9] = dimsConv7.c(); d[10] = dimsConv7.h(); d[11] = dimsConv7.w();
	    d[12] = dimsConv8.c(); d[13] = dimsConv8.h(); d[14] = dimsConv8.w();
	    d[15] = dimsConv9.c(); d[16] = dimsConv9.h(); d[17] = dimsConv9.w();
	}



   // void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

	void  configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
	{
	    dimsConv4_3 = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
	    dimsFc7 = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
	    dimsConv6 = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
	    dimsConv7 = DimsCHW{inputs[3].d[0], inputs[3].d[1], inputs[3].d[2]};
	    dimsConv8 = DimsCHW{inputs[4].d[0], inputs[4].d[1], inputs[4].d[2]};
	    dimsConv9 = DimsCHW{inputs[5].d[0], inputs[5].d[1], inputs[5].d[2]};
	}



 

protected:
    DimsCHW dimsConv4_3, dimsFc7, dimsConv6, dimsConv7, dimsConv8, dimsConv9;
    int inputs_size;
    int top_concat_axis;//top 层 concat后的维度
    int* bottom_concat_axis = new int[9];//记录每个bottom层concat维度的shape
    int* concat_input_size_ = new int[9];
    int* num_concats_ = new int[9];
    int _axis;
};

#endif

//SSD Reshape layer : shape{0,-1,21}
template<int OutC>
class Reshape : public IPlugin
{
public:
    Reshape() {}
    Reshape(const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override
    {
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
        return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }

    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};

//SSD Flatten layer
class FlattenLayer : public IPlugin
{
public:
    FlattenLayer(){}
    FlattenLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        return DimsCHW(_size, 1, 1);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};



  
struct Profiler : public IProfiler  
{  
    typedef std::pair<std::string, float> Record;  
    std::vector<Record> mProfile;  
  
    virtual void reportLayerTime(const char* layerName, float ms)  
    {  
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });  
  
        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));  
        else record->second += ms;  
    }  
  
    void printLayerTimes(const int TIMING_ITERATIONS)  
    {  
        float totalTime = 0;  
        for (size_t i = 0; i < mProfile.size(); i++)  
        {  
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);  
            totalTime += mProfile[i].second;  
        }  
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);  
    }  
};  
  
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {  
public:  
  
        // caffe parser plugin implementation  
        bool isPlugin(const char* name) override  
        {  
        return (!strcmp(name, "conv4_3_norm")  
            || !strcmp(name, "conv4_3_norm_mbox_conf_perm")  
            || !strcmp(name, "conv4_3_norm_mbox_conf_flat")  
            || !strcmp(name, "conv4_3_norm_mbox_loc_perm")  
            || !strcmp(name, "conv4_3_norm_mbox_loc_flat")  
            || !strcmp(name, "fc7_mbox_conf_perm")  
            || !strcmp(name, "fc7_mbox_conf_flat")  
            || !strcmp(name, "fc7_mbox_loc_perm")  
            || !strcmp(name, "fc7_mbox_loc_flat")  
            || !strcmp(name, "conv6_2_mbox_conf_perm")  
            || !strcmp(name, "conv6_2_mbox_conf_flat")  
            || !strcmp(name, "conv6_2_mbox_loc_perm")  
            || !strcmp(name, "conv6_2_mbox_loc_flat")  
            || !strcmp(name, "conv7_2_mbox_conf_perm")  
            || !strcmp(name, "conv7_2_mbox_conf_flat")  
            || !strcmp(name, "conv7_2_mbox_loc_perm")  
            || !strcmp(name, "conv7_2_mbox_loc_flat")  
            || !strcmp(name, "conv8_2_mbox_conf_perm")  
            || !strcmp(name, "conv8_2_mbox_conf_flat")  
            || !strcmp(name, "conv8_2_mbox_loc_perm")  
            || !strcmp(name, "conv8_2_mbox_loc_flat")  
            || !strcmp(name, "conv9_2_mbox_conf_perm")  
            || !strcmp(name, "conv9_2_mbox_conf_flat")  
            || !strcmp(name, "conv9_2_mbox_loc_perm")  
            || !strcmp(name, "conv9_2_mbox_loc_flat")  
            || !strcmp(name, "conv4_3_norm_mbox_priorbox")  
            || !strcmp(name, "fc7_mbox_priorbox")  
            || !strcmp(name, "conv6_2_mbox_priorbox")  
            || !strcmp(name, "conv7_2_mbox_priorbox")  
            || !strcmp(name, "conv8_2_mbox_priorbox")  
            || !strcmp(name, "conv9_2_mbox_priorbox")  
            || !strcmp(name, "mbox_conf_reshape")  
            || !strcmp(name, "mbox_conf_flatten")  
            || !strcmp(name, "mbox_loc")  
            || !strcmp(name, "mbox_conf")  
            || !strcmp(name, "mbox_priorbox")  
            || !strcmp(name, "detection_out")  
        ||  !strcmp(name,    "detection_out2")  
        );  
  
        }  
  
        virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override  
        {  
                // there's no way to pass parameters through from the model definition, so we have to define it here explicitly  
                if(!strcmp(layerName, "conv4_3_norm")){  
  
            //INvPlugin *   plugin::createSSDNormalizePlugin (const Weights *scales, bool acrossSpatial, bool channelShared, float eps)  
  
            _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(weights,false,false,1e-10);  
  
            return _nvPlugins.at(layerName);  
  
                }else if(!strcmp(layerName, "conv4_3_norm_mbox_loc_perm")  
            ||  !strcmp(layerName, "conv4_3_norm_mbox_conf_perm")  
            ||  !strcmp(layerName,"fc7_mbox_loc_perm")  
            ||  !strcmp(layerName,"fc7_mbox_conf_perm")  
            ||  !strcmp(layerName,"conv6_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv6_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv7_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv7_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv8_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv8_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv9_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv9_2_mbox_conf_perm")  
        ){  
  
            _nvPlugins[layerName] = plugin::createSSDPermutePlugin(Quadruple({0,2,3,1}));  
  
            return _nvPlugins.at(layerName);  
  
                } else if(!strcmp(layerName,"conv4_3_norm_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {30.0f};   
            float maxSize[1] = {60.0f};   
            float aspectRatios[1] = {2.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 1;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 8.0f;  
            params.stepW = 8.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"fc7_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {60.0f};   
            float maxSize[1] = {111.0f};   
            float aspectRatios[2] = {2.0f, 3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 16.0f;  
            params.stepW = 16.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv6_2_mbox_priorbox")){  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {111.0f};   
            float maxSize[1] = {162.0f};   
            float aspectRatios[2] = {2.0f, 3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 32.0f;  
            params.stepW = 32.0f;  
            params.offset = 5.0f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv7_2_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {162.0f};   
            float maxSize[1] = {213.0f};   
            float aspectRatios[2] = {2.0f, 3.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 2;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 64.0f;  
            params.stepW = 64.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv8_2_mbox_priorbox")){  
  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {213.0f};   
            float maxSize[1] = {264.0f};   
            float aspectRatios[1] = {2.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 1;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 100.0f;  
            params.stepW = 100.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"conv9_2_mbox_priorbox")){  
            plugin::PriorBoxParameters params = {0};  
            float minSize[1] = {264.0f};   
            float maxSize[1] = {315.0f};   
            float aspectRatios[1] = {2.0f};   
            params.minSize = (float*)minSize;  
            params.maxSize = (float*)maxSize;  
            params.aspectRatios = (float*)aspectRatios;  
            params.numMinSize = 1;  
            params.numMaxSize = 1;  
            params.numAspectRatios = 1;  
            params.flip = true;  
            params.clip = false;  
            params.variance[0] = 0.10000000149;  
            params.variance[1] = 0.10000000149;  
            params.variance[2] = 0.20000000298;  
            params.variance[3] = 0.20000000298;  
            params.imgH = 0;  
            params.imgW = 0;  
            params.stepH = 300.0f;  
            params.stepW = 300.0f;  
            params.offset = 0.5f;  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"detection_out")  
            ||!strcmp(layerName,"detection_out2")  
        ){  
            /*  
            bool    shareLocation  
            bool    varianceEncodedInTarget  
            int     backgroundLabelId  
            int     numClasses  
            int     topK  
            int     keepTopK  
            float   confidenceThreshold  
            float   nmsThreshold  
            CodeType_t  codeType  
            */  
            plugin::DetectionOutputParameters params = {0};  
            params.numClasses = 21;  
            params.shareLocation = true;  
            params.varianceEncodedInTarget = false;  
            params.backgroundLabelId = 0;  
            params.keepTopK = 200;  
            params.codeType = CENTER_SIZE;  
            params.nmsThreshold = 0.45;  
            params.topK = 400;  
            params.confidenceThreshold = 0.5;  
            _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin (params);  
            return _nvPlugins.at(layerName);  
        }else if (  
            !strcmp(layerName, "conv4_3_norm_mbox_conf_flat")  
            ||!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv9_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv9_2_mbox_loc_flat")  
            ||!strcmp(layerName,"mbox_conf_flatten")  
        ){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer());  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_reshape")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new Reshape<21>();  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_loc")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (1,false);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (1,false);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_priorbox")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (2,false);  
            return _nvPlugins.at(layerName);  
        }else {  
            assert(0);  
            return nullptr;  
        }  
    }  
  
    // deserialization plugin implementation  
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {                
        if(!strcmp(layerName, "conv4_3_norm"))  
        {  
            _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(  
            !strcmp(layerName, "conv4_3_norm_mbox_loc_perm")  
            || !strcmp(layerName, "conv4_3_norm_mbox_conf_perm")  
            || !strcmp(layerName,"fc7_mbox_loc_perm")  
            || !strcmp(layerName,"fc7_mbox_conf_perm")  
            || !strcmp(layerName,"conv6_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv6_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv7_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv7_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv8_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv8_2_mbox_conf_perm")  
            || !strcmp(layerName,"conv9_2_mbox_loc_perm")  
            || !strcmp(layerName,"conv9_2_mbox_conf_perm")  
        ){  
            _nvPlugins[layerName] = plugin::createSSDPermutePlugin(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
  
        }else if(!strcmp(layerName,"conv4_3_norm_mbox_priorbox")  
            || !strcmp(layerName,"fc7_mbox_priorbox")     
            || !strcmp(layerName,"conv6_2_mbox_priorbox")  
            || !strcmp(layerName,"conv7_2_mbox_priorbox")  
            || !strcmp(layerName,"conv8_2_mbox_priorbox")  
            || !strcmp(layerName,"conv9_2_mbox_priorbox")  
        ){  
            _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"detection_out")  
            || !strcmp(layerName,"detection_out2")  
            ){  
            _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf_reshape")){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)new Reshape<21>(serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if (  
            !strcmp(layerName, "conv4_3_norm_mbox_conf_flat")  
            ||!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_loc_flat")  
            ||!strcmp(layerName,"fc7_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv6_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv7_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv8_2_mbox_loc_flat")  
            ||!strcmp(layerName,"conv9_2_mbox_conf_flat")  
            ||!strcmp(layerName,"conv9_2_mbox_loc_flat")  
            ||!strcmp(layerName,"mbox_conf_flatten")  
        ){  
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer(serialData, serialLength));  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_loc")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_conf")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else if(!strcmp(layerName,"mbox_priorbox")){  
            _nvPlugins[layerName] = plugin::createConcatPlugin (serialData, serialLength);  
            return _nvPlugins.at(layerName);  
        }else{  
            assert(0);  
            return nullptr;  
        }  
    }  
  
  
    void destroyPlugin()  
    {  
        for (auto it=_nvPlugins.begin(); it!=_nvPlugins.end(); ++it){  
            it->second->destroy();  
            _nvPlugins.erase(it);  
        }  
    }  
  
  
private:  
  
        std::map<std::string, plugin::INvPlugin*> _nvPlugins;   
};  
  
  
  
#endif  

