# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

py = import('python').find_installation('python3', pure: false)

py.install_sources('_api.py', subdir: (cuda_wheel_dir))
py.install_sources('__init__.py', subdir: (cuda_wheel_dir))
py.install_sources('py.typed', subdir: (cuda_wheel_dir))
py.install_sources('logging.py', subdir: (cuda_wheel_dir))

py.install_sources(
  'device/__init__.py',
  subdir: join_paths(cuda_wheel_dir, 'device'),
)
if cute_device_enabled
  py.install_sources(
    [
      'device/cute/__init__.py',
      'device/cute/_bindings.py',
      'device/cute/_bitcode.py',
      'device/cute/_mlir.py',
      'device/cute/memory.py',
      'device/cute/ops.py',
      'device/cute/topology.py',
      'device/cute/types.py',
    ],
    subdir: join_paths(cuda_wheel_dir, 'device', 'cute'),
  )
endif
