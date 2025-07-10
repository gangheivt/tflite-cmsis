/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "micro_model_settings.h"
#include "models/ds_cnn_m_quantized.h"
#include "testdata/no_1000ms_audio_data.h"
#include "testdata/noise_1000ms_audio_data.h"
#include "testdata/silence_1000ms_audio_data.h"
#include "testdata/yes_1000ms_audio_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "mem_section.h"
#include "MFCC.hpp"

extern "C" int tf_main(int argc, char * argv [ ]);
#define main tf_main
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kArenaSize = 300*1024;

ALIGN(16)
uint8_t g_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];

constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<6>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<19>;

static tflite::MicroInterpreter * g_interpreter;
static MicroSpeechOpResolver op_resolver;
static int g_prediction_index;


TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

TfLiteStatus GenerateFeatures(const int16_t* data,
                              const size_t size,
                              float quantScale,
                              float quantOffset,
                              Features* features_output)
{
    std::vector<float> audioData(size);

     //DS-CNN model settings
    float SAMP_FREQ = kAudioSampleFrequency;
    int MFCC_WINDOW_LEN = kFeatureDurationMs*kAudioSampleFrequency/1000;
    int MFCC_WINDOW_STRIDE = kFeatureStrideMs*kAudioSampleFrequency/1000;
    int NUM_MFCC_FEATS = kFeatureSize;
    int NUM_MFCC_VECTORS = kFeatureCount;
    //todo: calc in pipeline and use in main
    int SAMPLES_PER_INFERENCE = NUM_MFCC_VECTORS * MFCC_WINDOW_STRIDE +
                                MFCC_WINDOW_LEN - MFCC_WINDOW_STRIDE; //16000
    float MEL_LO_FREQ = 20;
    float MEL_HI_FREQ = 4000;
    int NUM_FBANK_BIN = 40;

    MfccParams mfccParams(SAMP_FREQ,
                          NUM_FBANK_BIN,
                          MEL_LO_FREQ,
                          MEL_HI_FREQ,
                          NUM_MFCC_FEATS,
                          MFCC_WINDOW_LEN, false,
                          NUM_MFCC_VECTORS);

    std::unique_ptr<MFCC> mfccInst = std::make_unique<MFCC>(mfccParams);

    for (size_t i = 0; i < size; ++i) {
        // 将int16_t范围[-32768, 32767]归一化到float范围[-1.0, 1.0)
        audioData[i] = static_cast<float>(data[i]) / 32768.0f;
    }

    for (size_t i = 0; i < kFeatureCount; ++i) {
        auto mfccAudioData = std::vector<float>(
                audioData.data() + i*MFCC_WINDOW_STRIDE,  // 使用data()获取指针
                audioData.data() + i*MFCC_WINDOW_STRIDE + MFCC_WINDOW_LEN
        );

        //MicroPrintf("GenerateFeatures 3, %d, %f, %f",i, quantScale, quantOffset);
        auto mfcc = mfccInst->MfccComputeQuant<int8_t>(mfccAudioData, quantScale, quantOffset);

        for (size_t j = 0; j < mfcc.size() && j < 10; ++j) {
            (*features_output)[i][j] = static_cast<int8_t>(mfcc[j]);
        }
    }
    return kTfLiteOk;
}

TfLiteStatus LoadMicroSpeechModel(    )
{
    const tflite::Model* model = tflite::GetModel(g_micro_speech_quantized_model_data);

    TfLiteStatus r=RegisterOps(op_resolver);

    if (kTfLiteOk!=r)
        MicroPrintf("Could not Register OPs: %d.", r);
    else {
        tflite::MicroInterpreter *interpreter=new tflite::MicroInterpreter(model, op_resolver, g_arena, kArenaSize);
        r=interpreter->AllocateTensors();
        if (kTfLiteOk!=r) {
            MicroPrintf("Could not Allocate tensors: %d.", r);
        }
        else {
            MicroPrintf("kws model arena size = %u", interpreter->arena_used_bytes());
            g_interpreter=interpreter;
        }
    }

    return r;
}


TfLiteStatus UnloadModel(    )
{
    delete g_interpreter;
    g_interpreter = nullptr;

    return kTfLiteOk;
}

const char* PerformInference(const int16_t* audio_data, const size_t audio_data_size)
{
    g_prediction_index = -1;
    TfLiteTensor* input = g_interpreter->input(0);
    TfLiteTensor* output = g_interpreter->output(0);
    RT_ASSERT(output);
    TfLiteStatus r;

    Features m_features;
    GenerateFeatures(audio_data, audio_data_size, input->params.scale, input->params.zero_point, &m_features);

    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    std::copy_n(&m_features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(input));

    r=g_interpreter->Invoke();
    if (kTfLiteOk!=r) {
        MicroPrintf("Invoke Model failed: %d", r);
        return "Error";
    }
    else {
        // Dequantize output values
        float category_predictions[kCategoryCount];
        for (int i = 0; i < kCategoryCount; i++) {
            category_predictions[i] =
                (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
                output_scale;
#if 0
            MicroPrintf("  %.4f %s", static_cast<double>(category_predictions[i]),
                        kCategoryLabels[i]);
#endif
        }
        int prediction_index =
             std::distance(std::begin(category_predictions),
                            std::max_element(std::begin(category_predictions),
                                             std::end(category_predictions)));
        if (category_predictions[prediction_index] < kFeatureThreshold)
            prediction_index=kLabelUnknownIdx;
        g_prediction_index = prediction_index;
        return kCategoryLabels[prediction_index];
    }
}

TfLiteStatus PerformInferenceAndCheck(
    const int16_t* audio_data, const size_t audio_data_size, const char* expected_label)
{
    const char * label;

    MicroPrintf("MicroSpeech category predictions for <%s>", expected_label);
    label=PerformInference(audio_data, audio_data_size);
    TF_LITE_MICRO_EXPECT_STRING_EQ(expected_label,label);
    return kTfLiteOk;
}

TfLiteStatus TestAudioSample(const char* label, const int16_t* audio_data,
                             const size_t audio_data_size) {
  TF_LITE_ENSURE_STATUS(
      PerformInferenceAndCheck(audio_data, audio_data_size,  label));
  return kTfLiteOk;
}

}  // namespace

#define KWS_MIC_SUPPORT

#ifdef KWS_MIC_SUPPORT

#include <audio_server.h>
#include "webrtc/common_audio/vad/include/webrtc_vad.h"
extern "C" void mic_gain_decrease(int8_t db);


#define MIC_1000MS_DATA_BYTES                       (16000 * 2)
#define MAX_BYTES_APPENDED_WHILE_INFERENCE          (320 * 100)  //perform inference 1000ms data cost < 100ms
#define MOVE_WINDOWN_BYTES                          (MAX_BYTES_APPENDED_WHILE_INFERENCE/2) // move 50ms data
typedef enum VadState
{
    VAD_STATE_IDLE = 0,
    VAD_STATE_START = 1,
    VAD_STATE_GOT_VAD = 2,
    VAD_STATE_INFERENCE = 3,
    VAD_STATE_MOVE_WINDOW = 4
} VadStates;

static int test_using_mic = 1;

typedef struct
{
    audio_client_t  client;
    VadInst         *vad;
    uint8_t         data[MIC_1000MS_DATA_BYTES + MAX_BYTES_APPENDED_WHILE_INFERENCE];
    uint32_t        offset;
    VadStates       vad_state;
    rt_event_t      event;
} kws_handle_t;

static kws_handle_t *thiz;

static int kws_record_callback(audio_server_callback_cmt_t cmd, void *callback_userdata, uint32_t reserved)
{
    if (cmd == as_callback_cmd_data_coming)
    {
        audio_server_coming_data_t *p = (audio_server_coming_data_t *)reserved;
        RT_ASSERT(p->data_len == 320);

        if (thiz->vad_state == VAD_STATE_IDLE)
        {
            memcpy(&thiz->data[0], p->data, p->data_len);
            thiz->offset = p->data_len;
            thiz->vad_state = VAD_STATE_START;
            return 0;
        }
        if (thiz->vad_state == VAD_STATE_START)
        {
            int ret;
            ret = WebRtcVad_Process(thiz->vad, 16000, (int16_t*)p->data, p->data_len/2);
            if (ret)
            {
                MicroPrintf("voice start");
                memcpy(&thiz->data[thiz->offset], p->data, p->data_len);
                thiz->offset += p->data_len;
                thiz->vad_state = VAD_STATE_GOT_VAD;
            }
            else
            {
                //reserved 50ms data before vad detected
                memcpy(&thiz->data[0], &thiz->data[320], 320 * 4);
                memcpy(&thiz->data[320 * 4], p->data, 320);
                thiz->offset = 320 * 5;
            }
            return 0;
        }

        if (thiz->vad_state == VAD_STATE_GOT_VAD)
        {
            memcpy(&thiz->data[thiz->offset], p->data, p->data_len);
            thiz->offset += p->data_len;
            if (thiz->offset >= MIC_1000MS_DATA_BYTES)
            {
                thiz->vad_state = VAD_STATE_INFERENCE;
                rt_event_send(thiz->event, 1);
            }
            return 0;
        }

        if (thiz->vad_state == VAD_STATE_MOVE_WINDOW)
        {
            if (thiz->offset + p->data_len <= sizeof(thiz->data))
            {
                memcpy(&thiz->data[thiz->offset], p->data, p->data_len);
                thiz->offset += p->data_len;
            }
            if (thiz->offset >= MIC_1000MS_DATA_BYTES + MOVE_WINDOWN_BYTES)
            {
                memcpy(&thiz->data[0], &thiz->data[MOVE_WINDOWN_BYTES], MIC_1000MS_DATA_BYTES);
                thiz->offset = MIC_1000MS_DATA_BYTES;
                thiz->vad_state = VAD_STATE_INFERENCE;
                rt_event_send(thiz->event, 1);
            }
            return 0;
        }

        if (thiz->vad_state == VAD_STATE_INFERENCE)
        {
            if (thiz->offset + p->data_len <= sizeof(thiz->data))
            {
                memcpy(&thiz->data[thiz->offset], p->data, p->data_len);
                thiz->offset += p->data_len;
            }
            else
            {
                MicroPrintf("drop mic data while inference");
            }
            return 0;
        }

    }

    return 0;
}
static kws_handle_t * kws_open()
{
    int ret;
    RT_ASSERT(!thiz);

    LoadMicroSpeechModel();

    thiz = (kws_handle_t *)rt_malloc(sizeof(kws_handle_t));
    RT_ASSERT(thiz);
    memset(thiz, 0 , sizeof(kws_handle_t));
    thiz->offset = 0;
    thiz->event = rt_event_create("kws", RT_IPC_FLAG_FIFO);
    RT_ASSERT(thiz->event);
    ret = WebRtcVad_Create(&thiz->vad);
    RT_ASSERT(!ret);
    ret = WebRtcVad_Init(thiz->vad);
    RT_ASSERT(!ret);
    ret = WebRtcVad_set_mode(thiz->vad, 3); // 0 ~ 3
    RT_ASSERT(!ret);

    audio_parameter_t pa = {0};
    /*open audio record*/
    pa.write_bits_per_sample = 16;
    pa.write_channnel_num = 1;
    pa.write_samplerate = kAudioSampleFrequency;
    pa.write_cache_size = 4;
    pa.read_bits_per_sample = 16;
    pa.read_channnel_num = 1;
    pa.read_samplerate = kAudioSampleFrequency;
    pa.read_cache_size = 0;
    thiz->client = audio_open(AUDIO_TYPE_LOCAL_RECORD, AUDIO_RX, &pa, kws_record_callback, (void *)NULL);
    RT_ASSERT(thiz->client);
    MicroPrintf("kws_open done");

    mic_gain_decrease(4);
    return thiz;
}
static void kws_close()
{
    RT_ASSERT(thiz);
    audio_close(thiz->client);
    if (thiz->vad)
    {
        WebRtcVad_Free(thiz->vad);
        thiz->vad = NULL;
    }
    rt_event_delete(thiz->event);
    rt_free(thiz);
    thiz = NULL;
    UnloadModel();
}

extern "C" int tf_main(int argc, char* argv[])
{

    kws_open();
    MicroPrintf("please speaking\n");
    while (test_using_mic)
    {
        const char * label;
        rt_uint32_t evt = 0;
        rt_event_recv(thiz->event, 1, RT_EVENT_FLAG_OR | RT_EVENT_FLAG_CLEAR, RT_WAITING_FOREVER, &evt);
        label = PerformInference((const int16_t*)&thiz->data[0], MIC_1000MS_DATA_BYTES / 2);
        if (strcmp(label,"unknown")!=0 && strcmp(label,"silence")!=0)
            MicroPrintf("Result:%s", label);

        if (g_prediction_index != kLabelUnknownIdx)
        {
            MicroPrintf("match index=%d", g_prediction_index);
            memset(&thiz->data[0], 0, sizeof(thiz->data));
            thiz->vad_state = VAD_STATE_IDLE;
        }
        else
        {
            MicroPrintf("no match, move voice window");
#if 1
            thiz->vad_state = VAD_STATE_MOVE_WINDOW;
#else
            thiz->vad_state = VAD_STATE_IDLE;
#endif
        }
    }

    kws_close();
    return 0;
}

int kws_mic(int argc, char **argv)
{
    test_using_mic = !test_using_mic;
    return 0;
}

MSH_CMD_EXPORT_ALIAS(kws_mic, kws_mic, kws_mic);

#else

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Init) {
  LoadMicroSpeechModel();
}

TF_LITE_MICRO_TEST(NoTest) {
  TestAudioSample("no", g_no_1000ms_audio_data, g_no_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(YesTest) {
  TestAudioSample("yes", g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(SilenceTest) {
  TestAudioSample("silence", g_silence_1000ms_audio_data,
                  g_silence_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(NoiseTest) {
  TestAudioSample("unknown", g_noise_1000ms_audio_data,
                  g_noise_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(DeInit) {
  UnloadModel();
}

TF_LITE_MICRO_TESTS_END
#endif


