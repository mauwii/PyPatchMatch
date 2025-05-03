project('patchmatch', 'cpp',
  version : '1.0.2',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

opencv_dep = dependency('opencv4', required: true)

py = import('python').find_installation(pure: false)

csrc_files = files(
  'patchmatch/csrc/inpaint.cpp',
  'patchmatch/csrc/masked_image.cpp',
  'patchmatch/csrc/nnf.cpp',
  'patchmatch/csrc/pyinterface.cpp'
)


libpatchmatch = shared_library('patchmatch',
  csrc_files,
  install : true,
  install_dir: py.get_install_dir() / 'patchmatch',
  soversion : '0',
  dependencies: [opencv_dep])

py.install_sources(['patchmatch/__init__.py', 'patchmatch/patch_match.py'], subdir: 'patchmatch')

