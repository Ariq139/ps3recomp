#!/usr/bin/env python3
"""Static/semantic fixtures for the opt-in register promotion pass."""
import pickle
from register_regions import PureGprLine, promote_register_regions


def test_occurrence_provenance():
    a = PureGprLine("    ctx->gpr[3] = ctx->gpr[4] + 1;", "ppu", 0x1000, "addi")
    b = PureGprLine("    ctx->gpr[4] = ctx->gpr[3] + 1;", "ppu", 0x1004, "addi")
    unknown = str(a)
    assert type(unknown) is str and unknown == a
    assert promote_register_regions([unknown, b], "ppu") == [unknown, b]
    assert promote_register_regions([a, unknown, b], "ppu") == [a, unknown, b]
    promoted = promote_register_regions([a, b], "ppu")
    assert "uint64_t _rr3" in "\n".join(promoted)
    assert "_rr4 = _rr3 + 1;" in "\n".join(promoted)
    assert promote_register_regions([a, "loc_00001004:", b], "ppu") == [a, "loc_00001004:", b]
    # Text produced by an unrelated peephole cannot inherit old eligibility.
    assert promote_register_regions([a.replace(" + 1", " + 1"), b], "ppu") == [a, b]
    restored = pickle.loads(pickle.dumps([a, b]))
    assert restored[0].guest_pc == 0x1000 and restored[1].mnemonic == "addi"
    assert promote_register_regions(restored, "ppu") == promoted
    assert promote_register_regions(restored, "spu") == restored


if __name__ == "__main__":
    test_occurrence_provenance()
    print("PASS: register-region occurrence provenance")
