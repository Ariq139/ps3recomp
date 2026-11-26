#include <windows.h>
#include "../rsx_d3d12_shadow_border.h"
#include <stdio.h>
#include <string.h>

static int checks, failures;
#define CHECK(condition, name) do { \
    checks++; if (!(condition)) { failures++; fprintf(stderr, "FAIL %s\n", name); } \
} while (0)

static void check_case(D3D12_TEXTURE_ADDRESS_MODE u, D3D12_TEXTURE_ADDRESS_MODE v,
    D3D12_TEXTURE_ADDRESS_MODE w, D3D12_FILTER filter, int shadow,
    uint32_t border, int white)
{
    D3D12_SAMPLER_DESC actual = {0}, expected = {0};
    actual.AddressU = u;
    actual.AddressV = v;
    actual.AddressW = w;
    actual.Filter = filter;
    actual.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS;
    actual.MaxAnisotropy = 1;
    actual.MaxLOD = 8.0f;
    expected = actual;
    if (white)
        for (unsigned i = 0; i < 4; i++) expected.BorderColor[i] = 1.0f;
    rsx_d3d12_apply_shadow_border(&actual, shadow, border);
    CHECK(memcmp(&actual, &expected, sizeof actual) == 0,
          "only eligible border color changes");
}

int main(void)
{
    const D3D12_TEXTURE_ADDRESS_MODE b = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    const D3D12_TEXTURE_ADDRESS_MODE c = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    const D3D12_TEXTURE_ADDRESS_MODE w = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    const D3D12_FILTER normal = D3D12_FILTER_MIN_MAG_MIP_POINT;
    const D3D12_FILTER cmp = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    check_case(b, c, c, normal, 1, 0x00FFFFFFu, 1);
    check_case(w, b, c, cmp, 1, 0xFFFFFFFFu, 1);
    check_case(b, b, b, cmp, 1, 0x00FFFFFFu, 1);
    check_case(b, c, c, normal, 1, 0x00000000u, 0);
    check_case(b, c, c, normal, 1, 0xFF000000u, 0);
    check_case(b, c, c, normal, 1, 0x00808080u, 0);
    check_case(b, c, c, normal, 0, 0x00FFFFFFu, 0);
    check_case(w, c, b, normal, 1, 0x00FFFFFFu, 0);
    check_case(c, c, c, cmp, 1, 0x00FFFFFFu, 0);

    /* Sampler table identity compares complete descriptors. White and black
     * must remain distinct for the same texture/filter/address state. */
    D3D12_SAMPLER_DESC black = {0};
    black.AddressU = b;
    black.AddressV = c;
    black.AddressW = c;
    black.Filter = normal;
    D3D12_SAMPLER_DESC white = black;
    rsx_d3d12_apply_shadow_border(&white, 1, 0x00FFFFFFu);
    CHECK(memcmp(&white, &black, sizeof white) != 0, "border changes descriptor identity");
    D3D12_SAMPLER_DESC again = black;
    rsx_d3d12_apply_shadow_border(&again, 1, 0x00FFFFFFu);
    CHECK(memcmp(&white, &again, sizeof white) == 0, "stable descriptor identity");
    printf("D3D12 shadow border: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
