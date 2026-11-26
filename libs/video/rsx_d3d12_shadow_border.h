#ifndef PS3RECOMP_RSX_D3D12_SHADOW_BORDER_H
#define PS3RECOMP_RSX_D3D12_SHADOW_BORDER_H

#include <stdint.h>
#include <d3d12.h>

/* Supported 2D integer depth comparison textures use red-channel depth.
 * Match the established shared-engine far-plane endpoint (RGB24 all ones,
 * including 0x00FFFFFF). Only change samplers that can actually read a
 * border on a sampled axis. The comparison helper uses .r, so alpha is
 * irrelevant; use opaque white just like eng_border_class.
 *
 * General color/remap, intermediate depth border values, float depth and
 * embedded texture borders remain outside this deliberately narrow fix.
 */
static inline void rsx_d3d12_apply_shadow_border(D3D12_SAMPLER_DESC* sd,
    int supported_shadow, uint32_t border)
{
    if (!supported_shadow || (border & 0x00FFFFFFu) != 0x00FFFFFFu ||
        (sd->AddressU != D3D12_TEXTURE_ADDRESS_MODE_BORDER &&
         sd->AddressV != D3D12_TEXTURE_ADDRESS_MODE_BORDER))
        return;

    for (unsigned i = 0; i < 4; i++) sd->BorderColor[i] = 1.0f;
}

#endif
