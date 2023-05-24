// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OnnxruntimeModule.h"
#import "TensorHelper.h"

#import <Foundation/Foundation.h>
#import <React/RCTLog.h>
#import <React/RCTBridge+Private.h>
#import <React/RCTUtils.h>
#import <ReactCommon/RCTTurboModule.h>
#import <jsi/jsi.h>

#import "ThreadPool.h"

// Note: Using below syntax for including ort c api and ort extensions headers to resolve a compiling error happened
// in an expo react native ios app when ort extensions enabled (a redefinition error of multiple object types defined
// within ORT C API header). It's an edge case that compiler allows both ort c api headers to be included when #include
// syntax doesn't match. For the case when extensions not enabled, it still requires a onnxruntime prefix directory for
// searching paths. Also in general, it's a convention to use #include for C/C++ headers rather then #import. See:
// https://google.github.io/styleguide/objcguide.html#import-and-include
// https://microsoft.github.io/objc-guide/Headers/ImportAndInclude.html
#ifdef ORT_ENABLE_EXTENSIONS
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_extensions.h"
#else
#include "onnxruntime/onnxruntime_cxx_api.h"
#endif

@implementation OnnxruntimeModule

struct SessionInfo {
  std::unique_ptr<Ort::Session> session;
  std::vector<const char *> inputNames;
  std::vector<Ort::AllocatedStringPtr> inputNames_ptrs;
  std::vector<const char *> outputNames;
  std::vector<Ort::AllocatedStringPtr> outputNames_ptrs;
};

static Ort::Env *ortEnv = new Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");
static NSMutableDictionary *sessionMap = [NSMutableDictionary dictionary];
static Ort::AllocatorWithDefaultOptions ortAllocator;

static int nextSessionId = 0;
- (NSString *)getNextSessionKey {
  NSString *key = @(nextSessionId).stringValue;
  nextSessionId++;
  return key;
}

RCT_EXPORT_MODULE(Onnxruntime)

/**
 * React native binding API to load a model using given uri.
 *
 * @param modelPath a model file location. it's used as a key when multiple sessions are created, i.e. multiple models
 * are loaded.
 * @param options onnxruntime session options
 * @param resolve callback for returning output back to react native js
 * @param reject callback for returning an error back to react native js
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
RCT_EXPORT_METHOD(loadModel
                  : (NSString *)modelPath options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary *resultMap = [self loadModel:modelPath options:options];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to load model", nil);
  }
}

/**
 * React native binding API to load a model using BASE64 encoded model data string.
 *
 * @param modelData the BASE64 encoded model data string
 * @param options onnxruntime session options
 * @param resolve callback for returning output back to react native js
 * @param reject callback for returning an error back to react native js
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
RCT_EXPORT_METHOD(loadModelFromBase64EncodedBuffer
                  : (NSString *)modelDataBase64EncodedString options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSData *modelDataDecoded = [[NSData alloc] initWithBase64EncodedString:modelDataBase64EncodedString options:0];
    NSDictionary *resultMap = [self loadModelFromBuffer:modelDataDecoded options:options];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to load model from buffer", nil);
  }
}

/**
 * React native binding API to run a model using given uri.
 *
 * @param url a model path location given at loadModel()
 * @param input an input tensor
 * @param output an output names to be returned
 * @param options onnxruntime run options
 * @param resolve callback for returning an inference result back to react native js
 * @param reject callback for returning an error back to react native js
 */
RCT_EXPORT_METHOD(run
                  : (NSString *)url input
                  : (NSDictionary *)input output
                  : (NSArray *)output options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary *resultMap = [self run:url input:input output:output options:options];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to run model", nil);
  }
}

/**
 * Load a model using given model path.
 *
 * @param modelPath a model file location.
 * @param options onnxruntime session options.
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
- (NSDictionary *)loadModel:(NSString *)modelPath options:(NSDictionary *)options {
  return [self loadModelImpl:modelPath modelData:nil options:options];
}

/**
 * Load a model using given model data array
 *
 * @param modelData the model data buffer.
 * @param options onnxruntime session options
 */
- (NSDictionary *)loadModelFromBuffer:(NSData *)modelData options:(NSDictionary *)options {
  return [self loadModelImpl:@"" modelData:modelData options:options];
}

/**
 * Load model implementation method given either model data array or model path
 *
 * @param modelPath the model file location.
 * @param modelData the model data buffer.
 * @param options onnxruntime session options.
 */
- (NSDictionary *)loadModelImpl:(NSString *)modelPath modelData:(NSData *)modelData options:(NSDictionary *)options {
  SessionInfo *sessionInfo = nullptr;
  sessionInfo = new SessionInfo();
  Ort::SessionOptions sessionOptions = [self parseSessionOptions:options];

#ifdef ORT_ENABLE_EXTENSIONS
  Ort::ThrowOnError(RegisterCustomOps(sessionOptions, OrtGetApiBase()));
#endif

  if (modelData == nil) {
    sessionInfo->session.reset(new Ort::Session(*ortEnv, [modelPath UTF8String], sessionOptions));
  } else {
    NSUInteger dataLength = [modelData length];
    Byte *modelBytes = (Byte *)[modelData bytes];
    sessionInfo->session.reset(new Ort::Session(*ortEnv, modelBytes, (size_t)dataLength, sessionOptions));
  }

  sessionInfo->inputNames.reserve(sessionInfo->session->GetInputCount());
  for (size_t i = 0; i < sessionInfo->session->GetInputCount(); ++i) {
    auto inputName = sessionInfo->session->GetInputNameAllocated(i, ortAllocator);
    sessionInfo->inputNames.emplace_back(inputName.get());
    sessionInfo->inputNames_ptrs.emplace_back(std::move(inputName));
  }

  sessionInfo->outputNames.reserve(sessionInfo->session->GetOutputCount());
  for (size_t i = 0; i < sessionInfo->session->GetOutputCount(); ++i) {
    auto outputName = sessionInfo->session->GetOutputNameAllocated(i, ortAllocator);
    sessionInfo->outputNames.emplace_back(outputName.get());
    sessionInfo->outputNames_ptrs.emplace_back(std::move(outputName));
  }

  NSString *key = [self getNextSessionKey];
  NSValue *value = [NSValue valueWithPointer:(void *)sessionInfo];
  sessionMap[key] = value;

  NSMutableDictionary *resultMap = [NSMutableDictionary dictionary];
  resultMap[@"key"] = key;

  NSMutableArray *inputNames = [NSMutableArray array];
  for (auto inputName : sessionInfo->inputNames) {
    [inputNames addObject:[NSString stringWithCString:inputName encoding:NSUTF8StringEncoding]];
  }
  resultMap[@"inputNames"] = inputNames;

  NSMutableArray *outputNames = [NSMutableArray array];
  for (auto outputName : sessionInfo->outputNames) {
    [outputNames addObject:[NSString stringWithCString:outputName encoding:NSUTF8StringEncoding]];
  }
  resultMap[@"outputNames"] = outputNames;

  return resultMap;
}


//- (facebook::jsi::Object)loadModelImplJSI:(NSString *)modelPath
//                                    modelData:(NSData *)modelData
//                                    options:(facebook::jsi::Object)options {
//  SessionInfo *sessionInfo = nullptr;
//  sessionInfo = new SessionInfo();
//
//  // TODO
//  // Ort::SessionOptions sessionOptions = [self parseSessionOptionsJSI:options];
//
//
//}

/**
 * Run a model using given uri.
 *
 * @param url a model path location given at loadModel()
 * @param input an input tensor
 * @param output an output names to be returned
 * @param options onnxruntime run options
 */
- (NSDictionary *)run:(NSString *)url
                input:(NSDictionary *)input
               output:(NSArray *)output
              options:(NSDictionary *)options {
  NSValue *value = [sessionMap objectForKey:url];
  if (value == nil) {
    NSException *exception = [NSException exceptionWithName:@"onnxruntime"
                                                     reason:@"can't find onnxruntime session"
                                                   userInfo:nil];
    @throw exception;
  }
  SessionInfo *sessionInfo = (SessionInfo *)[value pointerValue];

  std::vector<Ort::Value> feeds;
  std::vector<Ort::MemoryAllocation> allocations;
  feeds.reserve(sessionInfo->inputNames.size());
  for (auto inputName : sessionInfo->inputNames) {
    NSDictionary *inputTensor = [input objectForKey:[NSString stringWithUTF8String:inputName]];
    if (inputTensor == nil) {
      NSException *exception = [NSException exceptionWithName:@"onnxruntime" reason:@"can't find input" userInfo:nil];
      @throw exception;
    }

    Ort::Value value = [TensorHelper createInputTensor:inputTensor ortAllocator:ortAllocator allocations:allocations];
    feeds.emplace_back(std::move(value));
  }

  std::vector<const char *> requestedOutputs;
  requestedOutputs.reserve(output.count);
  for (NSString *outputName : output) {
    requestedOutputs.emplace_back([outputName UTF8String]);
  }
  Ort::RunOptions runOptions = [self parseRunOptions:options];

  auto result =
      sessionInfo->session->Run(runOptions, sessionInfo->inputNames.data(), feeds.data(),
                                sessionInfo->inputNames.size(), requestedOutputs.data(), requestedOutputs.size());

  NSDictionary *resultMap = [TensorHelper createOutputTensor:requestedOutputs values:result];

  return resultMap;
}

static NSDictionary *graphOptimizationLevelTable = @{
  @"disabled" : @(ORT_DISABLE_ALL),
  @"basic" : @(ORT_ENABLE_BASIC),
  @"extended" : @(ORT_ENABLE_EXTENDED),
  @"all" : @(ORT_ENABLE_ALL)
};

static NSDictionary *executionModeTable = @{@"sequential" : @(ORT_SEQUENTIAL), @"parallel" : @(ORT_PARALLEL)};

- (Ort::SessionOptions)parseSessionOptions:(NSDictionary *)options {
  Ort::SessionOptions sessionOptions;

  if ([options objectForKey:@"intraOpNumThreads"]) {
    int intraOpNumThreads = [[options objectForKey:@"intraOpNumThreads"] intValue];
    if (intraOpNumThreads > 0 && intraOpNumThreads < INT_MAX) {
      sessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    }
  }

  if ([options objectForKey:@"interOpNumThreads"]) {
    int interOpNumThreads = [[options objectForKey:@"interOpNumThreads"] intValue];
    if (interOpNumThreads > 0 && interOpNumThreads < INT_MAX) {
      sessionOptions.SetInterOpNumThreads(interOpNumThreads);
    }
  }

  if ([options objectForKey:@"graphOptimizationLevel"]) {
    NSString *graphOptimizationLevel = [[options objectForKey:@"graphOptimizationLevel"] stringValue];
    if ([graphOptimizationLevelTable objectForKey:graphOptimizationLevel]) {
      sessionOptions.SetGraphOptimizationLevel(
          (GraphOptimizationLevel)[[graphOptimizationLevelTable objectForKey:graphOptimizationLevel] intValue]);
    }
  }

  if ([options objectForKey:@"enableCpuMemArena"]) {
    BOOL enableCpuMemArena = [[options objectForKey:@"enableCpuMemArena"] boolValue];
    if (enableCpuMemArena) {
      sessionOptions.EnableCpuMemArena();
    } else {
      sessionOptions.DisableCpuMemArena();
    }
  }

  if ([options objectForKey:@"enableMemPattern"]) {
    BOOL enableMemPattern = [[options objectForKey:@"enableMemPattern"] boolValue];
    if (enableMemPattern) {
      sessionOptions.EnableMemPattern();
    } else {
      sessionOptions.DisableMemPattern();
    }
  }

  if ([options objectForKey:@"executionMode"]) {
    NSString *executionMode = [[options objectForKey:@"executionMode"] stringValue];
    if ([executionModeTable objectForKey:executionMode]) {
      sessionOptions.SetExecutionMode((ExecutionMode)[[executionModeTable objectForKey:executionMode] intValue]);
    }
  }

  if ([options objectForKey:@"logId"]) {
    NSString *logId = [[options objectForKey:@"logId"] stringValue];
    sessionOptions.SetLogId([logId UTF8String]);
  }

  if ([options objectForKey:@"logSeverityLevel"]) {
    int logSeverityLevel = [[options objectForKey:@"logSeverityLevel"] intValue];
    sessionOptions.SetLogSeverityLevel(logSeverityLevel);
  }

  return sessionOptions;
}

- (Ort::RunOptions)parseRunOptions:(NSDictionary *)options {
  Ort::RunOptions runOptions;

  if ([options objectForKey:@"logSeverityLevel"]) {
    int logSeverityLevel = [[options objectForKey:@"logSeverityLevel"] intValue];
    runOptions.SetRunLogSeverityLevel(logSeverityLevel);
  }

  if ([options objectForKey:@"tag"]) {
    NSString *tag = [[options objectForKey:@"tag"] stringValue];
    runOptions.SetRunTag([tag UTF8String]);
  }

  return runOptions;
}

- (void)dealloc {
  NSEnumerator *iterator = [sessionMap keyEnumerator];
  while (NSString *key = [iterator nextObject]) {
    NSValue *value = [sessionMap objectForKey:key];
    SessionInfo *sessionInfo = (SessionInfo *)[value pointerValue];
    delete sessionInfo;
    sessionInfo = nullptr;
  }
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(install)
{
  NSLog(@"Installing ONNXRuntime Bindings...");
  RCTBridge* bridge = [RCTBridge currentBridge];
  RCTCxxBridge* cxxBridge = (RCTCxxBridge*)bridge;
  if (cxxBridge == nil) {
    return @false;
  }

  using namespace facebook;

  auto jsiRuntime = (jsi::Runtime*) cxxBridge.runtime;
  if (jsiRuntime == nil) {
    return @false;
  }

  auto& runtime = *jsiRuntime;
  auto pool = std::make_shared<ThreadPool>();
  auto callInvoker = bridge.jsCallInvoker;

  /**
    * Run a model using given uri.
    *
    * @param url a model path location given at loadModel()
    * @param input an input tensor
    * @param output an output names to be returned
    * @param options onnxruntime run options
    */
  auto run = jsi::Function::createFromHostFunction(runtime,
    jsi::PropNameID::forAscii(runtime, "onnxruntimeSessionRun"),
    4,
    [pool, callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) -> jsi::Value {
      if (count != 4) {
        throw jsi::JSError(runtime, "onnxruntimeSessionRun: Invalid number of args");
      }

      auto promise = runtime.global().getPropertyAsFunction(runtime, "Promise");
      return promise.callAsConstructor(runtime, jsi::Function::createFromHostFunction(runtime,
        jsi::PropNameID::forAscii(runtime, "executor"),
        2,
        [args, pool, callInvoker](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *pargs, size_t) -> jsi::Value
      {
        auto resolve = std::make_shared<jsi::Value>(runtime, pargs[0]);
        auto reject = std::make_shared<jsi::Value>(runtime, pargs[1]);

        auto url = args[0].getString(runtime).utf8(runtime);
        auto input = args[1].asObject(runtime);
        auto output = std::make_shared<jsi::Array>(args[2].asObject(runtime).asArray(runtime));
        auto options = args[3].asObject(runtime);

        NSString *urlString = [NSString stringWithUTF8String:url.c_str()];
        NSValue *value = [sessionMap objectForKey:urlString];
        if (value == nil) {
          throw jsi::JSError(runtime, "onnxruntimeSessionRun: can't find onnxruntime session");
        }
        SessionInfo *sessionInfo = (SessionInfo *)[value pointerValue];

        auto feeds = std::make_shared<std::vector<Ort::Value>>();
        std::vector<Ort::MemoryAllocation> allocations;
        feeds->reserve(sessionInfo->inputNames.size());
        for (auto inputName : sessionInfo->inputNames) {
          auto inputTensorProp = input.getProperty(runtime, inputName);
          if (inputTensorProp.isUndefined()) {
            throw jsi::JSError(runtime, "onnxInferenceRun: Invalid input tensor");
          }
          auto inputTensor = inputTensorProp.asObject(runtime);

          Ort::Value value = [TensorHelper createInputTensorJSI:runtime input:&inputTensor ortAllocator:ortAllocator allocations:allocations];
          feeds->emplace_back(std::move(value));
        }

        auto requestedOutputs = std::make_shared<std::vector<const char *>>();
        long outputCount = output->size(runtime);
        requestedOutputs->reserve(outputCount);
        for (int i = 0; i < outputCount; i++) {
          auto outputName = output->getValueAtIndex(runtime, i).asString(runtime).utf8(runtime);
          NSString *outputNameString = [NSString stringWithUTF8String:outputName.c_str()];
          requestedOutputs->emplace_back([outputNameString UTF8String]);
        }

        // Parse run options
        auto runOptions = std::make_shared<Ort::RunOptions>();
        if (options.hasProperty(runtime, "logSeverityLevel")) {
          int logSeverityLevel = options.getProperty(runtime, "logSeverityLevel").asNumber();
          runOptions->SetRunLogSeverityLevel(logSeverityLevel);
        }
        if (options.hasProperty(runtime, "tag")) {
          auto tag = options.getProperty(runtime, "tag").asString(runtime).utf8(runtime);
          runOptions->SetRunTag(tag.c_str());
        }

        pool->queueWork([callInvoker, &runtime, output, runOptions, sessionInfo, feeds, requestedOutputs, resolve, reject]() {
          auto result =
            sessionInfo->session->Run(*runOptions, sessionInfo->inputNames.data(), feeds->data(),
                                      sessionInfo->inputNames.size(), requestedOutputs->data(), requestedOutputs->size());

          auto resultPtr = std::make_shared<std::vector<Ort::Value>>(std::move(result));
          callInvoker->invokeAsync([&runtime, output, resultPtr, resolve]{
            auto requestedOutputs = std::make_shared<std::vector<const char *>>();
            long outputCount = output->asArray(runtime).size(runtime);
            requestedOutputs->reserve(outputCount);
            for (int i = 0; i < outputCount; i++) {
              auto outputName = output->getValueAtIndex(runtime, i).asString(runtime).utf8(runtime);
              NSString *outputNameString = [NSString stringWithUTF8String:outputName.c_str()];
              requestedOutputs->emplace_back([outputNameString UTF8String]);
            }
            facebook::jsi::Object resultMap = [TensorHelper createOutputTensorJSI:runtime outputNames:*requestedOutputs values:*resultPtr];
            resolve->asObject(runtime).asFunction(runtime).call(runtime, std::move(resultMap));
          });
        });
        return {};
      }));
    }
  );

  runtime.global().setProperty(runtime, "__onnxruntimeSessionRun", std::move(run));

  NSLog(@"Installed ONNXRuntime Bindings!");
  return @true;
}

@end
