# docker로 ai2thor cloudrendering(headless)돌릴때 생긴문제(vulkan, egl)

작성 일시: September 26, 2025 1:08 PM
태그: docker

report 버전.

ubuntu: 22.04

nvidia driver 

# 왜 이런 문제가 생기나

ai2thor를 cloudrendering(headless)로 돌리려면 정식으로 주입되는 vulkan의 nvidia.icd가 아니라 **egl**이라는걸 사용해야한다.

근데 어째서인지 이건 자동으로 읽지를 못한다. 

기본적으로 docker에서 vulkan을 받도록 해야 한다.

이때 내 우분투 버전과 적절한지 확인은 하자.

```docker
# --- Vulkan/GL 런타임 및 Unity 의존 라이브러리 설치 (LunarG 최신 버전 사용) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglvnd0 libgl1 libglx0 libegl1 \
    libglib2.0-0 libx11-6 libxext6 libxrandr2 libxi6 libxrender1 libxfixes3 libxcursor1 libvulkan-dev\
    libnss3 libasound2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Vulkan 로더 최신화 및 NVIDIA EGL ICD 설정  ---
RUN set -eux; \
    curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /usr/share/keyrings/lunarg-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/lunarg-archive-keyring.gpg] https://packages.lunarg.com/vulkan jammy main" > /etc/apt/sources.list.d/lunarg-vulkan.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libvulkan1 vulkan-tools vulkan-validationlayers; \
    rm -rf /var/lib/apt/lists/*

```

근데 이래도 안될거다.

```bash
ERROR: [Loader Message] Code 0 : vkCreateInstance: Found no drivers!
Cannot create Vulkan instance.
```

This problem is often caused by a faulty installation of the Vulkan driver or attempting to use a GPU that does not support Vulkan.
ERROR at ./vulkaninfo/./vulkaninfo.h:573:vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER

이때는 docker compose 의 environment에 VK_ICD_FILENAMES를 명시하는 방법이 있다.

```docker
services:
  ttp:
		...(기타 설정들)
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

```bash
WARNING: [Loader Message] Code 0 : loader_parse_icd_manifest: ICD JSON /usr/share/glvnd/egl_vendor.d/10_nvidia.json does not have an 'api_version' field. Skipping ICD JSON.
[Vulkan Loader] WARNING | DRIVER: loader_parse_icd_manifest: ICD JSON /usr/share/glvnd/egl_vendor.d/10_nvidia.json does not have an 'api_version' field. Skipping ICD JSON.
ERROR: [Loader Message] Code 0 : vkCreateInstance: Found no drivers!
[Vulkan Loader] ERROR | DRIVER: vkCreateInstance: Found no drivers!
Cannot create Vulkan instance.
This problem is often caused by a faulty installation of the Vulkan driver or attempting to use a GPU that does not support Vulkan.
ERROR at ./vulkaninfo/./vulkaninfo.h:573:vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER
```

WARNING: [Loader Message] Code 0 : loader_parse_icd_manifest: ICD JSON /usr/share/glvnd/egl_vendor.d/10_nvidia.json does not have an 'api_version' field. Skipping ICD JSON.

[https://stackoverflow.com/questions/74965945/vulkan-is-unable-to-detect-nvidia-gpu-from-within-a-docker-container-when-using](https://stackoverflow.com/questions/74965945/vulkan-is-unable-to-detect-nvidia-gpu-from-within-a-docker-container-when-using)

"api_version" : "1.3”