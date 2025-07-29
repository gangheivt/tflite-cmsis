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

// LED control includes
#ifdef KWS_MIC_SUPPORT
extern "C" {
#include "bf0_hal.h"
#include "drivers/rt_drv_pwm.h"
#include "drv_io.h"
#include "rtthread.h"
}

#define RGBLED_NAME "rgbled"
#define LED_GREEN_COLOR 0x000f00
#define LED_RED_COLOR   0x0f0000
#define LED_OFF_COLOR   0x000000

static struct rt_device *rgbled_device = NULL;

// LED color control function
static void kws_led_set_color(uint32_t color)
{
    if (rgbled_device)
    {
        struct rt_rgbled_configuration configuration;
        configuration.color_rgb = color;
        int ret = rt_device_control(rgbled_device, PWM_CMD_SET_COLOR, &configuration);
        if (ret != RT_EOK)
        {
            MicroPrintf("LED control failed with error: %d", ret);
        }
    }
    else
    {
        MicroPrintf("WARNING: rgbled_device is NULL, cannot set color 0x%06x", color);
    }
}

// LED initialization function
static void kws_led_init()
{
    MicroPrintf("Starting LED initialization...");
    
    // 配置外设LDO电源（仅黄山派开发板需要）
#ifdef BSP_USING_BOARD_SF32LB52_LCHSPI_ULP
    HAL_PMU_ConfigPeriLdo(PMU_PERI_LDO3_3V3, true, true);
    MicroPrintf("LDO power configured for SF32LB52-LCHSPI-ULP board");
#else
    MicroPrintf("Skip LDO configuration (not SF32LB52-LCHSPI-ULP board)");
#endif
    
    // 配置PA32引脚为PWM2通道1（按照ws2812示例的配置）
    HAL_PIN_Set(PAD_PA32, GPTIM2_CH1, PIN_NOPULL, 1);
    MicroPrintf("PIN PA32 configured for PWM2_CH1");
    
    // 等待设备稳定
    rt_thread_mdelay(100);
    
    // 查找RGB LED设备
    rgbled_device = rt_device_find(RGBLED_NAME);
    if (!rgbled_device)
    {
        MicroPrintf("ERROR: LED device '%s' not found!", RGBLED_NAME);
        MicroPrintf("Please check RGB LED configuration in proj.conf");
    }
    else
    {
        MicroPrintf("LED device '%s' found successfully", RGBLED_NAME);
        // 测试LED功能 - 启动时闪烁蓝灯表示初始化成功
        kws_led_set_color(0x00000f); // 蓝色
        rt_thread_mdelay(500);
        kws_led_set_color(LED_OFF_COLOR);
        rt_thread_mdelay(500);
        kws_led_set_color(0x00000f); // 蓝色
        rt_thread_mdelay(500);
        kws_led_set_color(LED_OFF_COLOR);
        MicroPrintf("LED initialization test completed");
    }
}

// LED control based on recognition result
static void kws_led_control(const char* label)
{
    if (!rgbled_device) return;
    
    // Check if it's one of the 12 valid keywords (excluding "silence" and "unknown")
    if (strcmp(label, "yes") == 0 || strcmp(label, "no") == 0 ||
        strcmp(label, "up") == 0 || strcmp(label, "down") == 0 ||
        strcmp(label, "left") == 0 || strcmp(label, "right") == 0 ||
        strcmp(label, "on") == 0 || strcmp(label, "off") == 0 ||
        strcmp(label, "stop") == 0 || strcmp(label, "go") == 0)
    {
        // Valid keyword detected - green light
        kws_led_set_color(LED_GREEN_COLOR);
        MicroPrintf("LED: Green (valid keyword: %s)", label);
    }
    else if (strcmp(label, "unknown") == 0)
    {
        // Unknown word detected - red light
        kws_led_set_color(LED_RED_COLOR);
        MicroPrintf("LED: Red (unknown word)");
    }
    else if (strcmp(label, "silence") == 0)
    {
        // Silence detected - turn off LED
        kws_led_set_color(LED_OFF_COLOR);
        // Don't print for silence to reduce noise
    }
    else
    {
        // Error case - turn off LED
        kws_led_set_color(LED_OFF_COLOR);
        MicroPrintf("LED: Off (error case: %s)", label);
    }
}
#endif

extern "C" int tf_main(int argc, char * argv [ ]);
#define main tf_main
#include "tensorflow/lite/micro/testing/micro_test.h"

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

static tflite::MicroInterpreter * g_interpreter;

#include "tensorflow/lite/micro/micro_profiler.h"
tflite::MicroProfiler profiler;

static MicroSpeechOpResolver op_resolver;


TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) 
{
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
        tflite::MicroInterpreter *interpreter=new tflite::MicroInterpreter(model, op_resolver, g_arena, kArenaSize, NULL, &profiler);
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
    TfLiteTensor* input = g_interpreter->input(0);
    TfLiteTensor* output = g_interpreter->output(0);
    TfLiteStatus r;

    Features m_features;
    GenerateFeatures(audio_data, audio_data_size, input->params.scale, input->params.zero_point, &m_features);
    
#ifndef KWS_MIC_SUPPORT    
    MicroPrintf("Feture generated");
#endif
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
#ifndef KWS_MIC_SUPPORT    
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

#ifdef KWS_MIC_SUPPORT

#include <audio_server.h>
#include "webrtc/common_audio/vad/include/webrtc_vad.h"
extern "C" void mic_gain_decrease(int8_t db);


#define MIC_1000MS_DATA_BYTES       (16000 * 2)

typedef struct
{
    audio_client_t  client;
    VadInst         *vad;
    uint8_t         data[MIC_1000MS_DATA_BYTES + 320];
    uint32_t        offset;
    int             is_vad_started;
    rt_event_t      event;
} kws_handle_t;

static kws_handle_t *thiz;

static int kws_record_callback(audio_server_callback_cmt_t cmd, void *callback_userdata, uint32_t reserved)
{
    if (cmd == as_callback_cmd_data_coming)
    {
        audio_server_coming_data_t *p = (audio_server_coming_data_t *)reserved;
        RT_ASSERT(p->data_len == 320);
        if (thiz->is_vad_started == 1)
        {
            memcpy(&thiz->data[thiz->offset], p->data, p->data_len);
            thiz->offset += p->data_len;
            if (thiz->offset >= MIC_1000MS_DATA_BYTES)
            {
                thiz->is_vad_started = 2;
                thiz->offset = 0;
                rt_event_send(thiz->event, 1);
            }
        }
        else if (thiz->is_vad_started == 2)
        {
            //kws is busy
        }
        else
        {
            int ret;
            ret = WebRtcVad_Process(thiz->vad, 16000, (int16_t*)p->data, p->data_len/2);
            if (ret)
            {
                MicroPrintf("voice start");
                memcpy(&thiz->data[thiz->offset], p->data, p->data_len);
                thiz->offset += p->data_len;
                thiz->is_vad_started = 1;
            }
            else
            {
                //reserved 100ms data before vad detected
                memcpy(&thiz->data[0], &thiz->data[320], 320 * 9);
                memcpy(&thiz->data[320 * 9], p->data, 320);
                thiz->offset = 320 * 10;
            }
        }
    }

    return 0;
}
static kws_handle_t * kws_open()
{
    int ret;
    RT_ASSERT(!thiz);
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
}

extern "C" int tf_main(int argc, char* argv[]) 
{
    LoadMicroSpeechModel();

    // Initialize LED for keyword indication
    kws_led_init();

    kws_open();
    MicroPrintf("please speaking\n");
    while (1)
    {
        const char * label;
        rt_uint32_t evt = 0;
        rt_event_recv(thiz->event, 1, RT_EVENT_FLAG_OR | RT_EVENT_FLAG_CLEAR, RT_WAITING_FOREVER, &evt);
        label=PerformInference((const int16_t*)&thiz->data[0], MIC_1000MS_DATA_BYTES / 2);
        
        // Control LED based on recognition result
        kws_led_control(label);
        
        if (strcmp(label,"unknown")!=0 && strcmp(label,"silence")!=0 )
            MicroPrintf("Result:%s", label);
        memset(&thiz->data[0], 0, MIC_1000MS_DATA_BYTES);
        thiz->is_vad_started = 0;
    }
    kws_close();
}    

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

profiler.LogTicksPerTagCsv();

TF_LITE_MICRO_TEST(DeInit) {
  UnloadModel();
}

TF_LITE_MICRO_TESTS_END
#endif


