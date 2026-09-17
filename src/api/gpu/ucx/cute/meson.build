# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cute_llvm_major = 20
cute_llvm_path = get_option('cute_llvm_path')
if cute_llvm_path == ''
  cute_clang = find_program('clang-20', 'clang', native: true, required: false)
  cute_opt = find_program('opt-20', 'opt', native: true, required: false)
  cute_llvm_dis = find_program('llvm-dis-20', 'llvm-dis', native: true, required: false)
  cute_llvm_as = find_program('llvm-as-20', 'llvm-as', native: true, required: false)
else
  cute_clang = find_program(join_paths(cute_llvm_path, 'clang'), native: true, required: false)
  cute_opt = find_program(join_paths(cute_llvm_path, 'opt'), native: true, required: false)
  cute_llvm_dis = find_program(join_paths(cute_llvm_path, 'llvm-dis'), native: true, required: false)
  cute_llvm_as = find_program(join_paths(cute_llvm_path, 'llvm-as'), native: true, required: false)
endif

cute_tools = [cute_clang, cute_opt, cute_llvm_dis, cute_llvm_as]
cute_tools_found = true
cute_llvm_versions_match = true
foreach tool : cute_tools
  if not tool.found()
    cute_tools_found = false
  elif not (tool.version().version_compare('>=' + cute_llvm_major.to_string()) and
            tool.version().version_compare('<' + (cute_llvm_major + 1).to_string()))
    cute_llvm_versions_match = false
  endif
endforeach

cute_build_enabled = cute_tools_found and cute_llvm_versions_match
if not cute_build_enabled
  reason = not cute_tools_found \
    ? 'clang, opt, llvm-dis, and llvm-as are all required' \
    : 'all LLVM tools must have major version ' + cute_llvm_major.to_string()
  if get_option('build_cute_device').enabled()
    error('cannot build CuTe DSL device bitcode: ' + reason)
  endif
  message('CuTe DSL device bitcode disabled: ' + reason)
else
  py_for_bitcode = import('python').find_installation('python3', pure: false)
  cute_python_install_dir = join_paths(
    py_for_bitcode.get_install_dir(),
    cuda_wheel_dir,
    'device',
    'cute',
  )
  bitcode_builder = files('../../../../../contrib/build_device_bitcode.py')
  bitcode_source = files('nixl_device_cute.cu')
  bitcode_command_base = [
    py_for_bitcode,
    bitcode_builder,
    '--source', '@INPUT@',
    '--output', '@OUTPUT0@',
    '--manifest', '@OUTPUT1@',
    '--depfile', '@DEPFILE@',
    '--clang', cute_clang,
    '--opt', cute_opt,
    '--llvm-dis', cute_llvm_dis,
    '--llvm-as', cute_llvm_as,
    '--llvm-major', cute_llvm_major.to_string(),
    '--nixl-version', meson.project_version(),
    '--include', join_paths(meson.project_source_root(), 'src/api/cpp'),
    '--include', join_paths(meson.project_source_root(), 'src/api/device'),
    '--include', join_paths(meson.project_source_root(), 'contrib/clang_compat'),
  ]
  if get_option('cute_device_validation')
    bitcode_command_base += ['--device-validation']
  endif

  cute_cuda_path = get_option('cute_cuda_path')
  if cute_cuda_path == '' and get_option('cudapath_inc') != ''
    cute_cuda_path = join_paths(get_option('cudapath_inc'), '..')
  endif
  if cute_cuda_path != ''
    if not cute_cuda_path.startswith('/')
      cute_cuda_path = join_paths(meson.project_source_root(), cute_cuda_path)
    endif
    bitcode_command_base += ['--cuda-home', cute_cuda_path]
  endif

  cute_ucx_path = get_option('ucx_path')
  if cute_ucx_path != ''
    if not cute_ucx_path.startswith('/')
      cute_ucx_path = join_paths(meson.project_source_root(), cute_ucx_path)
    endif
    bitcode_command_base += ['--ucx-path', cute_ucx_path]
  endif

  cute_arch_option = get_option('cute_bitcode_arch').strip()
  cute_bitcode_archs = []
  if cute_arch_option == 'auto'
    foreach selected_arch : selected_cuda_archs
      if selected_arch not in cute_bitcode_archs
        cute_bitcode_archs += [selected_arch]
      endif
    endforeach
  else
    foreach requested_arch : cute_arch_option.split(',')
      requested_arch = requested_arch.strip()
      if requested_arch == ''
        continue
      endif
      if requested_arch.startswith('sm_')
        requested_arch = requested_arch.replace('sm_', '')
      endif
      normalized_arch = 'sm_' + requested_arch.to_int().to_string()
      if normalized_arch not in selected_cuda_archs
        error('CuTe bitcode target ' + normalized_arch +
              ' is not present in nixl_cuda_arch_list (' +
              ', '.join(selected_cuda_archs) + ')')
      endif
      if normalized_arch not in cute_bitcode_archs
        cute_bitcode_archs += [normalized_arch]
      endif
    endforeach
  endif
  if cute_bitcode_archs.length() == 0
    error('cute_bitcode_arch selected no CUDA architectures')
  endif

  # Explicit one-architecture source builds retain the historical filenames.
  # Multi-architecture and auto builds use unambiguous per-SM pairs so runtime
  # selection can never fall through to bitcode compiled for another device.
  use_arch_suffix = cute_arch_option == 'auto' or cute_bitcode_archs.length() > 1
  cute_device_artifacts = []
  foreach cute_arch : cute_bitcode_archs
    if use_arch_suffix
      bitcode_output = 'libnixl_device_' + cute_arch + '.bc'
      manifest_output = 'nixl_device_abi_' + cute_arch + '.json'
      depfile_output = 'libnixl_device_' + cute_arch + '.d'
      bitcode_target_name = 'nixl_cute_device_bitcode_' + cute_arch
    else
      bitcode_output = 'libnixl_device.bc'
      manifest_output = 'nixl_device_abi.json'
      depfile_output = 'libnixl_device.d'
      bitcode_target_name = 'nixl_cute_device_bitcode'
    endif
    bitcode_command = bitcode_command_base + ['--cuda-arch', cute_arch]
    cute_device_artifacts += [custom_target(
      bitcode_target_name,
      input: bitcode_source,
      output: [bitcode_output, manifest_output],
      command: bitcode_command,
      depfile: depfile_output,
      depend_files: [
        files('../../../device/gpu/nixl_device.cuh'),
        files('../../../device/gpu/device_types.cuh'),
        files('../../../device/gpu/impl/device_dispatch.cuh'),
        files('../../../device/gpu/impl/ucx/device_ucx.cuh'),
        files('nixl_device_cute.cuh'),
        bitcode_builder,
      ],
      build_by_default: true,
      install: true,
      install_dir: cute_python_install_dir,
    )]
  endforeach
  if use_arch_suffix
    alias_target('nixl_cute_device_bitcode', cute_device_artifacts)
  endif
  cute_device_enabled = true
endif

summary({
  'CuTe device bitcode': cute_build_enabled,
  'CuTe LLVM major': cute_llvm_major,
  'CuTe bitcode arch': get_option('cute_bitcode_arch'),
  'CuTe device validation': get_option('cute_device_validation'),
  'CuTe CUDA path': get_option('cute_cuda_path'),
}, section: 'CuTe DSL', bool_yn: true)
