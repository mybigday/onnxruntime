// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "TensorHelper.h"
#import <Foundation/Foundation.h>
#import <jsi/jsi.h>

@implementation TensorHelper

/**
 * Supported tensor data type
 */
NSString *const JsTensorTypeBool = @"bool";
NSString *const JsTensorTypeUnsignedByte = @"uint8";
NSString *const JsTensorTypeByte = @"int8";
NSString *const JsTensorTypeShort = @"int16";
NSString *const JsTensorTypeInt = @"int32";
NSString *const JsTensorTypeLong = @"int64";
NSString *const JsTensorTypeFloat = @"float32";
NSString *const JsTensorTypeDouble = @"float64";
NSString *const JsTensorTypeString = @"string";


/**
 * It creates an input tensor from a map passed by react native js.
 * 'data' must be a string type as data is encoded as base64. It first decodes it and creates a tensor.
 */
+ (Ort::Value)createInputTensor:(NSDictionary *)input
                   ortAllocator:(OrtAllocator *)ortAllocator
                    allocations:(std::vector<Ort::MemoryAllocation> &)allocatons {
  // shape
  NSArray *dimsArray = [input objectForKey:@"dims"];
  std::vector<int64_t> dims;
  dims.reserve(dimsArray.count);
  for (NSNumber *dim in dimsArray) {
    dims.emplace_back([dim longLongValue]);
  }

  // type
  ONNXTensorElementDataType tensorType = [self getOnnxTensorType:[input objectForKey:@"type"]];

  // data
  if (tensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    NSArray *values = [input objectForKey:@"data"];
    auto inputTensor =
        Ort::Value::CreateTensor(ortAllocator, dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    size_t index = 0;
    for (NSString *value in values) {
      inputTensor.FillStringTensorElement([value UTF8String], index++);
    }
    return inputTensor;
  } else {
    NSString *data = [input objectForKey:@"data"];
    NSData *buffer = [[NSData alloc] initWithBase64EncodedString:data options:0];
    Ort::Value inputTensor = [self createInputTensor:tensorType
                                                dims:dims
                                              buffer:buffer
                                        ortAllocator:ortAllocator
                                         allocations:allocatons];
    return inputTensor;
  }
}

// copy createInputTensor but with jsi input
/**
 * It creates an input tensor from a map passed by react native js.
 * 'data' must be a string type as data is encoded as base64. It first decodes it and creates a tensor.
 */
+ (Ort::Value)createInputTensorJSI:(facebook::jsi::Runtime &)runtime
                   input:(const facebook::jsi::Object *)input
                   ortAllocator:(OrtAllocator *)ortAllocator
                    allocations:(std::vector<Ort::MemoryAllocation> &)allocatons {

  // shape
  facebook::jsi::Array dimsArray = input->getProperty(runtime, "dims").asObject(runtime).asArray(runtime);
  std::vector<int64_t> dims;
  dims.reserve(dimsArray.size(runtime));
  for (size_t i = 0; i < dimsArray.size(runtime); i++) {
    auto dim = dimsArray.getValueAtIndex(runtime, i).asNumber();
    dims.emplace_back(dim);
  }

  // type
  auto type = input->getProperty(runtime, "type").asString(runtime).utf8(runtime);
  NSString *typeString = [NSString stringWithUTF8String:type.c_str()];
  ONNXTensorElementDataType tensorType = [self getOnnxTensorType:typeString];

  // data
  if (tensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    facebook::jsi::Array values = input->getProperty(runtime, "data").asObject(runtime).asArray(runtime);
    auto inputTensor =
        Ort::Value::CreateTensor(ortAllocator, dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    size_t index = 0;
    for (size_t i = 0; i < values.size(runtime); i++) {
      auto value = values.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime);
      inputTensor.FillStringTensorElement(value.c_str(), index++);
    }
    return inputTensor;
  } else {
    // data change to array buffer
    auto obj = input->getProperty(runtime, "data").asObject(runtime);
    if (!obj.isArrayBuffer(runtime)) {
      throw facebook::jsi::JSError(runtime, "data must be an ArrayBuffer");
    }
    facebook::jsi::ArrayBuffer buffer = obj.getArrayBuffer(runtime);
    // if (buffer.size(runtime) == 0) {
    //   throw facebook::jsi::JSError(runtime, "data must not be empty");
    // }
    NSData *bufferData = [NSData dataWithBytesNoCopy:buffer.data(runtime) length:buffer.size(runtime) freeWhenDone:NO];
    Ort::Value inputTensor = [self createInputTensor:tensorType
                                                dims:dims
                                              buffer:bufferData
                                        ortAllocator:ortAllocator
                                         allocations:allocatons];
    return inputTensor;
  }
}

/**
 * It creates an output map from an output tensor.
 * a data array is encoded as base64 string.
 */
+ (NSDictionary *)createOutputTensor:(const std::vector<const char *> &)outputNames
                              values:(const std::vector<Ort::Value> &)values {
  if (outputNames.size() != values.size()) {
    NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                     reason:@"output name and tensor count mismatched"
                                                   userInfo:nil];
    @throw exception;
  }

  NSMutableDictionary *outputTensorMap = [NSMutableDictionary dictionary];

  for (size_t i = 0; i < outputNames.size(); ++i) {
    const auto outputName = outputNames[i];
    const Ort::Value &value = values[i];

    if (!value.IsTensor()) {
      NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                       reason:@"only tensor type is supported"
                                                     userInfo:nil];
      @throw exception;
    }

    NSMutableDictionary *outputTensor = [NSMutableDictionary dictionary];

    // dims
    NSMutableArray *outputDims = [NSMutableArray array];
    auto dims = value.GetTensorTypeAndShapeInfo().GetShape();
    for (auto dim : dims) {
      [outputDims addObject:[NSNumber numberWithLongLong:dim]];
    }
    outputTensor[@"dims"] = outputDims;

    // type
    outputTensor[@"type"] = [self getJsTensorType:value.GetTensorTypeAndShapeInfo().GetElementType()];

    // data
    if (value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      NSMutableArray *buffer = [NSMutableArray array];
      for (NSInteger i = 0; i < value.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        size_t elementLength = value.GetStringTensorElementLength(i);
        std::string element(elementLength, '\0');
        value.GetStringTensorElement(elementLength, i, (void *)element.data());
        [buffer addObject:[NSString stringWithUTF8String:element.data()]];
      }
      outputTensor[@"data"] = buffer;
    } else {
      NSString *data = [self createOutputTensor:value];
      outputTensor[@"data"] = data;
    }

    outputTensorMap[[NSString stringWithUTF8String:outputName]] = outputTensor;
  }

  return outputTensorMap;
}

+ (facebook::jsi::Object)createOutputTensorJSI:(facebook::jsi::Runtime &)runtime
                                        outputNames:(const std::vector<const char *> &)outputNames
                                        values:(const std::vector<Ort::Value> &)values {
  if (outputNames.size() != values.size()) {
    throw facebook::jsi::JSError(runtime, "output name and tensor count mismatched");
  }

  facebook::jsi::Object outputTensorMap(runtime);

  for (size_t i = 0; i < outputNames.size(); ++i) {
    const auto outputName = outputNames[i];
    const Ort::Value &value = values[i];

    if (!value.IsTensor()) {
      throw facebook::jsi::JSError(runtime, "only tensor type is supported");
    }

    facebook::jsi::Object outputTensor(runtime);

    // dims
    auto dims = value.GetTensorTypeAndShapeInfo().GetShape();
    facebook::jsi::Array outputDims(runtime, dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      // NOTE: onnxruntime-common js only supported safe integer dim
      outputDims.setValueAtIndex(runtime, i, (int) dims[i]);
    }
    outputTensor.setProperty(runtime, "dims", std::move(outputDims));

    // type
    NSString *jsTensorType = [self getJsTensorType:value.GetTensorTypeAndShapeInfo().GetElementType()];
    outputTensor.setProperty(runtime, "type",
                             facebook::jsi::String::createFromUtf8(runtime, [jsTensorType UTF8String]));

    // data
    if (value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      facebook::jsi::Array buffer(runtime, value.GetTensorTypeAndShapeInfo().GetElementCount());
      for (size_t i = 0; i < value.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        size_t elementLength = value.GetStringTensorElementLength(i);
        std::string element(elementLength, '\0');
        value.GetStringTensorElement(elementLength, i, (void *)element.data());
        buffer.setValueAtIndex(runtime, i, facebook::jsi::String::createFromUtf8(runtime, element));
      }
      outputTensor.setProperty(runtime, "data", std::move(buffer));
    } else {
      NSData *data = [self createOutputTensorJSI:value];
        // TODO: Use no-copy ArrayBuffer?
      facebook::jsi::Function arrayBufferCtor = runtime.global().getPropertyAsFunction(runtime, "ArrayBuffer");
      facebook::jsi::Object o = arrayBufferCtor.callAsConstructor(runtime, (int)data.length).getObject(runtime);
      facebook::jsi::ArrayBuffer buf = o.getArrayBuffer(runtime);
      memcpy(buf.data(runtime), data.bytes, data.length);
      outputTensor.setProperty(runtime, "data", buf);
    }

    outputTensorMap.setProperty(runtime, outputName, outputTensor);
  }
  return outputTensorMap;
}


template <typename T>
static Ort::Value createInputTensorT(OrtAllocator *ortAllocator, const std::vector<int64_t> &dims, NSData *buffer,
                                     std::vector<Ort::MemoryAllocation> &allocations) {
  T *dataBuffer = static_cast<T *>(ortAllocator->Alloc(ortAllocator, [buffer length]));
  allocations.emplace_back(ortAllocator, dataBuffer, [buffer length]);
  memcpy(static_cast<void *>(dataBuffer), [buffer bytes], [buffer length]);

  return Ort::Value::CreateTensor<T>(ortAllocator->Info(ortAllocator), dataBuffer, buffer.length / sizeof(T),
                                     dims.data(), dims.size());
}

+ (Ort::Value)createInputTensor:(ONNXTensorElementDataType)tensorType
                           dims:(const std::vector<int64_t> &)dims
                         buffer:(NSData *)buffer
                   ortAllocator:(OrtAllocator *)ortAllocator
                    allocations:(std::vector<Ort::MemoryAllocation> &)allocations {
  switch (tensorType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return createInputTensorT<float_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return createInputTensorT<uint8_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return createInputTensorT<int8_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return createInputTensorT<int16_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return createInputTensorT<int32_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return createInputTensorT<int64_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return createInputTensorT<bool>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return createInputTensorT<double_t>(ortAllocator, dims, buffer, allocations);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  default: {
    NSException *exception = [NSException exceptionWithName:@"create input tensor"
                                                     reason:@"unsupported tensor type"
                                                   userInfo:nil];
    @throw exception;
  }
  }
}

template <typename T> static NSString *createOutputTensorT(const Ort::Value &tensor) {
  const auto data = tensor.GetTensorData<T>();
  NSData *buffer = [NSData dataWithBytesNoCopy:(void *)data
                                        length:tensor.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(T)
                                  freeWhenDone:false];
  return [buffer base64EncodedStringWithOptions:0];
}

+ (NSString *)createOutputTensor:(const Ort::Value &)tensor {
  ONNXTensorElementDataType tensorType = tensor.GetTensorTypeAndShapeInfo().GetElementType();

  switch (tensorType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return createOutputTensorT<float_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return createOutputTensorT<uint8_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return createOutputTensorT<int8_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return createOutputTensorT<int16_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return createOutputTensorT<int32_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return createOutputTensorT<int64_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return createOutputTensorT<bool>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return createOutputTensorT<double_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  default: {
    NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                     reason:@"unsupported tensor type"
                                                   userInfo:nil];
    @throw exception;
  }
  }
}


template <typename T> static NSData *createOutputTensorTJSI(const Ort::Value &tensor) {
  const auto data = tensor.GetTensorData<T>();
  NSData *buffer = [NSData dataWithBytesNoCopy:(void *)data
                                        length:tensor.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(T)
                                  freeWhenDone:false];
  return buffer;
}

+ (NSData *)createOutputTensorJSI:(const Ort::Value &)tensor {
  ONNXTensorElementDataType tensorType = tensor.GetTensorTypeAndShapeInfo().GetElementType();

  switch (tensorType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return createOutputTensorTJSI<float_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return createOutputTensorTJSI<uint8_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return createOutputTensorTJSI<int8_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return createOutputTensorTJSI<int16_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return createOutputTensorTJSI<int32_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return createOutputTensorTJSI<int64_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return createOutputTensorTJSI<bool>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return createOutputTensorTJSI<double_t>(tensor);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  default: {
    NSException *exception = [NSException exceptionWithName:@"create output tensor"
                                                     reason:@"unsupported tensor type"
                                                   userInfo:nil];
    @throw exception;
  }
  }
}

NSDictionary *JsTensorTypeToOnnxTensorTypeMap;
NSDictionary *OnnxTensorTypeToJsTensorTypeMap;

+ (void)initialize {
  JsTensorTypeToOnnxTensorTypeMap = @{
    JsTensorTypeFloat : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
    JsTensorTypeUnsignedByte : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
    JsTensorTypeByte : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
    JsTensorTypeShort : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
    JsTensorTypeInt : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
    JsTensorTypeLong : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
    JsTensorTypeString : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
    JsTensorTypeBool : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
    JsTensorTypeDouble : @(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
  };

  OnnxTensorTypeToJsTensorTypeMap = @{
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) : JsTensorTypeFloat,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) : JsTensorTypeUnsignedByte,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) : JsTensorTypeByte,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) : JsTensorTypeShort,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) : JsTensorTypeInt,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) : JsTensorTypeLong,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) : JsTensorTypeString,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) : JsTensorTypeBool,
    @(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) : JsTensorTypeDouble
  };
}

+ (ONNXTensorElementDataType)getOnnxTensorType:(const NSString *)type {
  if ([JsTensorTypeToOnnxTensorTypeMap objectForKey:type]) {
    return (ONNXTensorElementDataType)[JsTensorTypeToOnnxTensorTypeMap[type] intValue];
  } else {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

+ (NSString *)getJsTensorType:(ONNXTensorElementDataType)type {
  if ([OnnxTensorTypeToJsTensorTypeMap objectForKey:@(type)]) {
    return OnnxTensorTypeToJsTensorTypeMap[@(type)];
  } else {
    return @"undefined";
  }
}

@end
