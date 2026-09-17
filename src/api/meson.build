# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

subdir('cpp')

cute_device_artifacts = []
cute_device_enabled = false
if not get_option('build_cute_device').disabled()
  if ucx_gpu_device_api_available
    subdir('gpu/ucx/cute')
  elif get_option('build_cute_device').enabled()
    error('build_cute_device requires the UCX GPU Device API')
  else
    message('CuTe DSL device bitcode disabled: UCX GPU Device API is unavailable')
  endif
endif

subdir('python')
subdir('device')
