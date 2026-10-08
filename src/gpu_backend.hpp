// Added for the Windows ROCm fork: backend selection shared by the motion and text models.
// See docs/windows-rocm.md.
#pragma once

#include "profile.hpp"

#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace kimodo::detail {

// KIMODO_BACKEND=cpu selects the CPU reference path for parity triage.
inline bool gpu_backend_requested() noexcept {
    const char *choice = std::getenv("KIMODO_BACKEND");
    return !choice || std::string_view(choice) != "cpu";
}

// Opens the first device of the GGML GPU backend compiled into this build
// (HIP/ROCm in the Windows ROCm build).  Returns nullptr when no device is
// usable, which leaves the caller on its CPU fallback.
inline ggml_backend_t create_gpu_backend() {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) return nullptr;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend && profile_enabled()) {
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        std::fprintf(stderr, "profile backend device=\"%s\" vram_total_mib=%zu vram_free_mib=%zu\n",
                     ggml_backend_dev_description(device), total_bytes >> 20, free_bytes >> 20);
    }
    return backend;
}

} // namespace kimodo::detail
