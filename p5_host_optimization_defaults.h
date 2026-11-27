#ifndef PS3RECOMP_P5_HOST_OPTIMIZATION_DEFAULTS_H
#define PS3RECOMP_P5_HOST_OPTIMIZATION_DEFAULTS_H

#include <stdio.h>
#include <stdlib.h>

static inline void p5_set_env_default(const char* name, const char* value)
{
    if (getenv(name)) return;
#ifdef _WIN32
    if (_putenv_s(name, value) != 0)
#else
    if (setenv(name, value, 0) != 0)
#endif
        fprintf(stderr, "[host-opt] unable to set %s\n", name);
}

static inline const char* p5_env_value(const char* name)
{
    const char* value = getenv(name);
    return value ? value : "unset";
}

static inline void p5_set_host_optimization_defaults(void)
{
#if defined(PS3RECOMP_P5_HOST_OPTIMIZATION_DEFAULTS)
    /* Environment-provided values always win, including an explicit "0". */
    p5_set_env_default("SPURS_JQ_SHARE_IMAGES", "1");
    p5_set_env_default("RSX_D3D12_SAMPLER_KEY_CACHE", "1");
    p5_set_env_default("RSX_D3D12_INDEX_CACHE", "1");
    /* These only take effect if the optional live-draw renderer is selected. */
    p5_set_env_default("RSX_LIVE_INDEX_CACHE", "1");
    p5_set_env_default("RSX_LIVE_FP_CONSTANT_REUSE", "1");
#endif
    fprintf(stderr,
        "[host-opt] SPURS_JQ_SHARE_IMAGES=%s RSX_D3D12_SAMPLER_KEY_CACHE=%s "
        "RSX_D3D12_INDEX_CACHE=%s RSX_LIVE_DRAW=%s RSX_LIVE_INDEX_CACHE=%s "
        "RSX_LIVE_FP_CONSTANT_REUSE=%s\n",
        p5_env_value("SPURS_JQ_SHARE_IMAGES"),
        p5_env_value("RSX_D3D12_SAMPLER_KEY_CACHE"),
        p5_env_value("RSX_D3D12_INDEX_CACHE"),
        p5_env_value("RSX_LIVE_DRAW"),
        p5_env_value("RSX_LIVE_INDEX_CACHE"),
        p5_env_value("RSX_LIVE_FP_CONSTANT_REUSE"));
}

#endif
