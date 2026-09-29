{
  description = "ComfyUI ROCm 6.3 dev shell (gfx906)";

  inputs = {
    pins.url = "path:/etc/nixos/pins";
    nixpkgs.follows = "pins/nixpkgs";
    #gpu-platform.url = "path:/home/luna/nix/gpu-platform";
  };

  outputs =
    {
      self,
	  pins,
      nixpkgs,
      #gpu-platform,
    }:
    let
      system = "x86_64-linux";

      overlays = [
        (final: prev: {
          rocmPackages = prev.rocmPackages.overrideScope (
            rfinal: rprev: {
              amdgpuTargets = [ "gfx906" ];
            }
          );
        })
      ];

      pkgs = import nixpkgs {
        inherit system overlays;
      };



      lib = pkgs.lib;

      # ROCm package set
      rocm = pkgs.rocmPackages;

      rocmLibs = [
        {
          env = "ROCM_RUNTIME";
          drv = rocm.rocm-runtime;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "ROCM_ROOT";
          drv = rocm.clr;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIP_PATH";
          drv = rocm.clr;
          addBin = false;
          addLib = false;
          addInclude = false;
        }

        {
          env = "HIPRAND_ROOT";
          drv = rocm.hiprand;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "ROCRAND_ROOT";
          drv = rocm.rocrand;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "ROCBLAS_ROOT";
          drv = rocm.rocblas;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIPBLAS_ROOT";
          drv = rocm.hipblas;
          addBin = true;
          addLib = true;
          addInclude = true;
        }

        {
          env = "HIPBLAS_COMMON_ROOT";
          drv = rocm.hipblas-common;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIPBLASLT_ROOT";
          drv = rocm.hipblaslt;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIP_COMMON_ROOT";
          drv = rocm.hip-common;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "MIOPEN_ROOT";
          drv = rocm.miopen;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIPFFT_ROOT";
          drv = rocm.hipfft;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIPSPARSE_ROOT";
          drv = rocm.hipsparse;
          addBin = true;
          addLib = true;
          addInclude = true;
        }

        {
          env = "ROCPRIM_ROOT";
          drv = rocm.rocprim;
          addBin = false;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIPCUB_ROOT";
          drv = rocm.hipcub;
          addBin = false;
          addLib = true;
          addInclude = true;
        }
        {
          env = "ROCTHRUST_ROOT";
          drv = rocm.rocthrust;
          addBin = false;
          addLib = true;
          addInclude = true;
        }
        {
          env = "HIPSOLVER_ROOT";
          drv = rocm.hipsolver;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "ROCSOLVER_ROOT";
          drv = rocm.rocsolver;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "ROCM_SMI_ROOT";
          drv = rocm.rocm-smi;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # amdsmi
        {
          env = "AMDSMI_ROOT";
          drv = rocm.amdsmi;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        {
          env = "RCCL_ROOT";
          drv = rocm.rccl;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # half
        {
          env = "HALF_ROOT";
          drv = rocm.half;
          addBin = true;
          addLib = true;
          addInclude = true;
        }


        # rocm-device-libs
        {
          env = "ROCM_DEVICE_LIBS_ROOT";
          drv = rocm.rocm-device-libs;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # rocalution
        {
          env = "ROCALUTION_ROOT";
          drv = rocm.rocalution;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # tensile
        {
          env = "TENSILE_ROOT";
          drv = rocm.tensile;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # rocwmma
        {
          env = "ROCWMMAROOT";
          drv = rocm.rocwmma;
          addBin = true;
          addLib = true;
          addInclude = true;
        }

        # rocsparse
        {
          env = "ROCSPARSE_ROOT";
          drv = rocm.rocsparse;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # rocmlir
        {
          env = "ROCMLIR_ROOT";
          drv = rocm.rocmlir;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # rocm-toolchain
        {
          env = "ROCM_TOOLCHAIN_ROOT";
          drv = rocm.rocm-toolchain;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # roctracer
        {
          env = "ROCTRACER_ROOT";
          drv = rocm.roctracer;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # rocprofiler
        {
          env = "ROCPROFILER_ROOT";
          drv = rocm.rocprofiler;
          addBin = true;
          addLib = true;
          addInclude = true;
        }
        # rocprofiler-sdk (kineto needs librocprofiler-sdk.so specifically)
        {
          env = "ROCPROFILER_SDK_ROOT"; # or whatever your builder script actually keys off — check next
          drv = rocm.rocprofiler-sdk.dev;
          addBin = false;
          addLib = true;
          addInclude = true;
        }
		        # rocprofiler-sdk (kineto needs librocprofiler-sdk.so specifically)
        {
          env = "ROCPROFILER_SDK_ROOT"; # or whatever your builder script actually keys off — check next
          drv = rocm.rocprofiler-sdk;
          addBin = false;
          addLib = true;
          addInclude = true;
        }
      ];

      rocmDrvList = map (pkg: pkg.drv) rocmLibs;

      runtimeLibs = [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.zstd
        pkgs.libglvnd
        pkgs.libxcb
        pkgs.glib
        pkgs.gcc
        pkgs.vulkan-loader
        pkgs.gfortran
        pkgs.fftw
        pkgs.openssl
        pkgs.pkg-config
        pkgs.protobuf
        pkgs.ninja
        pkgs.libGL
        pkgs.libGLU
        pkgs.mesa
      ]
      ++ rocmDrvList;

      rocmEnv = builtins.listToAttrs (
        map (pkg: {
          name = pkg.env;
          value = "${pkg.drv}";
        }) rocmLibs
      );

      rocmIncludePath = lib.makeSearchPath "include" (
        map (pkg: pkg.drv) (lib.filter (pkg: pkg.addInclude) rocmLibs)
      );

      # torch's CPU BLAS calls pass 32-bit ints, but nixpkgs' default openblas is
      # ILP64 on x86_64 (INTERFACE64) -- every fp32 CPU matmul segfaults with it
      # (e.g. Qwen3-VL rope freqs on a CPU text encoder). The self-built wheel's
      # RUNPATH still names an ILP64 openblas; listing this LP64 build first on
      # LD_LIBRARY_PATH overrides that without rebuilding torch on BLAS updates.
      cpuBlasLibs = [ pkgs.openblasCompat ];

      cpuBlasDebug = ''echo "CPU BLAS (LP64)  = ${pkgs.openblasCompat}/lib"'';

      ld_library_path = lib.makeLibraryPath (cpuBlasLibs ++ runtimeLibs);

      rocmLibraryPath = lib.makeLibraryPath (map (pkg: pkg.drv) (lib.filter (pkg: pkg.addLib) rocmLibs));

      rocmBinPath = lib.makeSearchPath "bin" (map (pkg: pkg.drv) (lib.filter (pkg: pkg.addBin) rocmLibs));

      debugPackages = lib.concatMapStringsSep "\n" (pkg: ''
        echo "======================================"
        echo "${pkg.env}"
        echo "  root    : ${pkg.drv}"
        [ -d "${pkg.drv}/bin" ]     && echo "  bin     : ${pkg.drv}/bin"
        [ -d "${pkg.drv}/include" ] && echo "  include : ${pkg.drv}/include"
        [ -d "${pkg.drv}/lib" ]     && echo "  lib     : ${pkg.drv}/lib"
      '') rocmLibs;
    in
    {
      devShells.${system}.default =
        (pkgs.mkShell.override {
          stdenv = rocm.llvm.rocmClangStdenv;
        })
          (
            {
              buildInputs = runtimeLibs;

              inputsFrom = [
                # gpu-platform.devShells.full
              ];

              nativeBuildInputs = runtimeLibs;

              packages = [
                pkgs.git
                pkgs.git-lfs
                pkgs.ccache
                pkgs.cmakeWithGui
                rocm.rocminfo
                pkgs.vulkan-tools
                pkgs.uv
                pkgs.ninja

                pkgs.python313
                pkgs.python313Packages.setuptools
                pkgs.python313Packages.wheel

                pkgs.zlib
                pkgs.libxml2
                pkgs.libGL

              ]
              ++ runtimeLibs;

              #
              # ROCm
              #

              USE_ROCM = "1";
              #USE_SYSTEM_ROCM = "1";

              PYTORCH_ROCM_ARCH = "gfx906";
              HCC_AMDGPU_TARGET = "gfx906";
              CMAKE_HIP_ARCHITECTURES = "gfx906";
              # AOTRITON_INSTALL_FROM_SOURCE = 1;
              USE_NVSHMEM = "0";

              CC = "clang";
              CXX = "clang++";

              CMAKE_GENERATOR = "Ninja";
              BUILD_TEST = "0";
              BUILD_TESTING = "0";
              USE_EIGEN_SPARSE = "0"; # note: you had "SPARCE" — typo, silently ignored either way
              USE_FBGEMM = "0";
              USE_FLASH_ATTENTION = "0";
              USE_NATIVE_ARCH = "1";
              USE_PRECOMPILED_HEADERS = "1";
              USE_ROCM_CK_GEMM = "0";
              USE_ROCM_CK_SDPA = "0";
              # USE_SYSTEM_LIBS = "1";
              USE_C10D_GLOO = "0";
              USE_C10D_NCCL = "0";
              USE_NCCL = "0";
              USE_DISTRIBUTED = "0";
              TORCH_USE_IWYU = "1";
              USE_NUMA = "0";
              USE_CUDA = "0";
              USE_XPU = "0";
              USE_ROCTX = "0";
              USE_KINETO = "1";
              #AOTRITON_INSTALL_FROM_SOURCE = "1";
			  USE_ROCM_KERNEL_ASSERT = "1";
              USE_ROCTRACER = "1";
              HIPCC_COMPILE_FLAGS_APPEND = "-Wno-error=c++11-narrowing-const-reference";
              EXTRA_CAFFE2_CMAKE_FLAGS = "-DHIP_HIPCC_FLAGS=-Wno-c++11-narrowing-const-reference";
              #
              # Python
              #
              CXXFLAGS = "-include cmath";
              PYTHON_EXECUTABLE = "${pkgs.python313}/bin/python3.13";
              Python3_EXECUTABLE = "$PYTHON_EXECUTABLE";

              PYTHON_INCLUDE_DIR = "${pkgs.python313}/include/python3.13";
              Python3_INCLUDE_DIR = "$PYTHON_INCLUDE_DIR";

              # These may be removable.
              # Keep them until PyTorch builds reliably.
              PYTHON_LIBRARY = "${pkgs.python313}/lib/libpython3.13.so";
              Python3_LIBRARY = "$PYTHON_LIBRARY";

              #
              # Misc build configuration
              #

              LLVM_TARGETS_TO_BUILD = "AMDGPU";
              CMAKE_BUILD_WITH_INSTALL_RPATH = "ON";

              ROC_ENABLE_PRE_VEGA = "1";
              HIP_FORCE_DEV_KERNARG = "1";

              PKG_CONFIG_PATH = "${pkgs.zlib.dev}/lib/pkgconfig:" + "${pkgs.libxml2.dev}/lib/pkgconfig";

              CMAKE_PREFIX_PATH = "${pkgs.zlib.dev}:" + "${pkgs.libxml2.dev}";

              ZLIB_LIBRARY = "${pkgs.zlib.dev}/lib/libz.so";
              ZLIB_INCLUDE_DIR = "${pkgs.zlib.dev}/include";
              CCACHE_DIR = "/media/luna/cache/ccache";
              #
              # shell-only setup
              #
              LD_LIBRARY_PATH = "${ld_library_path}";
              shellHook = ''
                hash -r
                export ROCM_PATH="$ROCM_ROOT"
                export PATH="${pkgs.python313}/bin:${rocmBinPath}:${rocm.clr}/llvm/bin:$PATH"

                export CPATH="$PYTHON_INCLUDE_DIR:${rocmIncludePath}"
                export C_INCLUDE_PATH="$CPATH"
                export CPLUS_INCLUDE_PATH="$CPATH"

                export CFLAGS="-I$PYTHON_INCLUDE_DIR $CFLAGS"
                export CPPFLAGS="$CFLAGS"

                HIP_VERSION=$(
                    hipcc --version |
                    sed -nE 's/.*HIP version: ([0-9.]+).*/\1/p'
                )

                echo "ComfyUI ROCm $HIP_VERSION shell ($PYTORCH_ROCM_ARCH)"

                echo "hipcc: $(command -v hipcc)"
                hipcc --version

                echo
                rocminfo | head

                if [ -d venv ]; then
                    source venv/bin/activate
                    # nixpkgs' python setup hooks put every ROCm python
                    # package's transitive deps (numpy, pandas, psutil, ...,
                    # pulled in for amdsmi/tensile/rocprofiler-sdk) on
                    # PYTHONPATH -- those entries land before the venv's own
                    # site-packages in sys.path (a python interpreter-
                    # startup ordering; venv activation alone doesn't
                    # override it), so the venv's pip-installed numpy/etc.
                    # get silently shadowed by nix's build of them. Prepend
                    # the venv back on top so it wins for anything it
                    # actually provides, while amdsmi/tensile/rocprofiler-sdk
                    # (nix-only, not in the venv) stay reachable as a
                    # fallback further down PYTHONPATH.
                    export PYTHONPATH="$PWD/venv/lib/python3.13/site-packages:$PYTHONPATH"
                fi

                echo
                echo "ROCM_ROOT        = $ROCM_ROOT"
                echo "HIPBLASLT_ROOT   = $HIPBLASLT_ROOT"
                echo "MIOPEN_ROOT      = $MIOPEN_ROOT"
                ${cpuBlasDebug}
                echo
                echo "Ready."


              '';
            }
            // rocmEnv
          );
    };
}
