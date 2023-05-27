// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from '@fugood/onnxruntime-common';
import {registerBackend} from '@fugood/onnxruntime-common';
import {onnxruntimeBackend} from './backend';

registerBackend('cpu', onnxruntimeBackend, 1);
