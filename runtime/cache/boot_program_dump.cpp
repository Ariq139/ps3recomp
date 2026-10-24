/* Startup program discovery. File-format fields are decoded locally; the PS3
 * SDK is reference material, never a build dependency. */
#include "boot_program_dump.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using byte = uint8_t;
constexpr uint32_t scan_version = 3;
constexpr size_t ls_size = 0x40000;
constexpr size_t max_unpacked = 256u * 1024u * 1024u;
bool disabled(const char* name) {
    const char* v = std::getenv(name);
    return v && (!std::strcmp(v, "0") || !std::strcmp(v, "false"));
}
uint32_t be32(const byte* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | p[3];
}
uint16_t be16(const byte* p) { return uint16_t((p[0] << 8) | p[1]); }
uint32_t le32(const byte* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t hash(const byte* p, size_t n, uint64_t h = 14695981039346656037ull) {
    while (n--) { h ^= *p++; h *= 1099511628211ull; }
    return h;
}
std::string hex(uint64_t n) {
    std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << n;
    return out.str();
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) {
            char b[7]; std::snprintf(b, sizeof(b), "\\u%04x", unsigned(c)); out += b;
        } else out += char(c);
    }
    return out + '"';
}
fs::path cache_root() {
    if (const char* p = std::getenv("PS3RECOMP_CACHE_DIR"); p && *p) return fs::path(p);
#ifdef _WIN32
    // Match D3D12/SPU's Win32 environment lookup. UCRT getenv(LOCALAPPDATA)
    // can return the packaged-app redirected location instead.
    std::vector<wchar_t> local(32768);
    DWORD count = GetEnvironmentVariableW(L"LOCALAPPDATA", local.data(), DWORD(local.size()));
    if (count && count < local.size()) return fs::path(local.data()) / "ps3recomp/cache";
    count = GetTempPathW(DWORD(local.size()), local.data());
    if (count && count < local.size()) return fs::path(local.data()) / "ps3recomp/cache";
    throw std::runtime_error("cache root unavailable");
#else
    if (const char* p = std::getenv("XDG_CACHE_HOME"); p && *p)
        return fs::path(p) / "ps3recomp/cache";
    if (const char* p = std::getenv("HOME"); p && *p)
        return fs::path(p) / ".cache/ps3recomp/cache";
    return fs::temp_directory_path() / "ps3recomp/cache";
#endif
}
struct MappedFile {
    const byte* data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    explicit MappedFile(const fs::path& path) {
        file = CreateFileW(path.c_str(), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        LARGE_INTEGER n{};
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &n) ||
            n.QuadPart <= 0 || uint64_t(n.QuadPart) > SIZE_MAX) return;
        size = size_t(n.QuadPart);
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping) data = static_cast<const byte*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    }
    ~MappedFile() {
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
#else
    int file = -1;
    explicit MappedFile(const fs::path& path) {
        file = open(path.c_str(), O_RDONLY);
        struct stat st{};
        if (file < 0 || fstat(file, &st) || st.st_size <= 0 ||
            uint64_t(st.st_size) > SIZE_MAX) return;
        size = size_t(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0);
        if (p != MAP_FAILED) data = static_cast<const byte*>(p);
    }
    ~MappedFile() { if (data) munmap(const_cast<byte*>(data), size); if (file >= 0) close(file); }
#endif
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

/* Return an exact ELF file extent only after validating every table/range. */
size_t spu_elf_extent(const byte* p, size_t n) {
    if (n < 52 || std::memcmp(p, "\x7f" "ELF\x01\x02\x01", 7) || be16(p + 18) != 23)
        return 0;
    const size_t phoff = be32(p + 28), shoff = be32(p + 32);
    const size_t ps = be16(p + 42), pn = be16(p + 44);
    const size_t ss = be16(p + 46), sn = be16(p + 48);
    if (!pn || ps < 32 || (sn && ss < 40) ||
        phoff > n || pn > (n - phoff) / ps ||
        (sn && (shoff > n || sn > (n - shoff) / ss))) return 0;
    size_t end = std::max(size_t(52), phoff + ps * pn);
    if (sn) end = std::max(end, shoff + ss * sn);
    for (size_t i = 0; i < pn; ++i) {
        const byte* h = p + phoff + i * ps;
        size_t off = be32(h + 4), len = be32(h + 16);
        if (off > n || len > n - off) return 0;
        end = std::max(end, off + len);
    }
    for (size_t i = 0; i < sn; ++i) {
        const byte* h = p + shoff + i * ss;
        if (be32(h + 4) == 8) continue;
        size_t off = be32(h + 16), len = be32(h + 20);
        if (off > n || len > n - off) return 0;
        end = std::max(end, off + len);
    }
    return end <= 16u * 1024u * 1024u ? end : 0;
}
bool guid(const byte* p, size_t n) {
    if (n < 16) return false;
    for (unsigned i = 0; i < 16; i += 4)
        if ((p[i] != 0x42 && p[i] != 0x43) || (p[i+3] != 2 && p[i+3] != 0x82)) return false;
    return true;
}

size_t cgb_extent(const byte* p, size_t n) {
    if (n < 32 || std::memcmp(p, "CGB\0", 4) || be16(p + 4) != 0 ||
        p[10] > 1 || (p[11] & ~7u) || ((p[11] & 4) && !(p[11] & 2))) return 0;
    const size_t code = be16(p + 8);
    if (!code || code % 16 || code > n - 32) return 0;
    size_t end = 32 + code;
    for (unsigned block : {1u, 2u, 4u}) {
        if (!(p[11] & block)) continue;
        const size_t header = block == 2 ? 6 : 4;
        if (end > n || n - end < header) return 0;
        size_t size = be16(p + end), count = be16(p + end + 2);
        if (size < header || size > n - end) return 0;
        if (block == 1 && count > (size - header) / 18) return 0;
        if (block == 2 && (count > (size - header) / 8 ||
            be16(p + end + 4) > (size - header - count * 8) / 2)) return 0;
        if (block == 4 && count > (size - header) / 8) return 0;
        end += size;
    }
    return end;
}

/* CRILAYLA stores a 256-byte prefix verbatim and encodes the remaining bytes
 * backwards. All offsets and backreferences stay within this member. */
std::vector<byte> unpack_crilayla(const byte* p, size_t n) {
    if (n < 16 || std::memcmp(p, "CRILAYLA", 8)) return {};
    const size_t output = size_t(le32(p + 8)) + 256;
    const size_t packed = le32(p + 12);
    if (output > max_unpacked || packed > n - 16 || n - 16 - packed < 256) return {};
    std::vector<byte> out(output);
    std::memcpy(out.data(), p + 16 + packed, 256);
    size_t pos = 16 + packed, dst = output;
    unsigned bits = 0, available = 0;
    auto take = [&](unsigned count) {
        unsigned value = 0;
        while (count--) {
            if (!available) {
                if (pos <= 16) throw std::runtime_error("truncated CRILAYLA bits");
                bits = p[--pos]; available = 8;
            }
            value = (value << 1) | ((bits >> --available) & 1u);
        }
        return value;
    };
    while (dst > 256) {
        if (!take(1)) out[--dst] = byte(take(8));
        else {
            const size_t distance = take(13) + 3u;
            size_t length = 3;
            bool extend = true;
            for (unsigned width : {2u, 3u, 5u, 8u}) {
                unsigned v = take(width); length += v;
                if (v != (1u << width) - 1u) { extend = false; break; }
            }
            while (extend) {
                unsigned v = take(8); length += v; extend = v == 255;
                if (length > dst - 256) throw std::runtime_error("CRILAYLA length overflow");
            }
            if (length > dst - 256 || distance > output - dst)
                throw std::runtime_error("invalid CRILAYLA backreference");
            while (length--) { --dst; out[dst] = out[dst + distance]; }
        }
    }
    return out;
}

struct Collector {
    fs::path root;
    bool spu = !disabled("SPU_DUMP_JOB_IMAGES"), shaders = !disabled("RSX_DUMP_SHADER_IMAGES");
    std::ofstream report;
    std::vector<std::pair<fs::path, uintmax_t>> outputs;
    uint64_t shader_count = 0, spu_count = 0, regions = 0, compressed = 0, rejected = 0;
    fs::path save(const char* folder, const char* prefix, const char* ext,
                  const byte* data, size_t size) {
        std::string key = hex(hash(data, size)) + hex(hash(data, size, 7809847782465536322ull));
        fs::path dest = root / folder / (std::string(prefix) + key + ext);
        fs::create_directories(dest.parent_path());
        bool equal = false;
        if (fs::exists(dest) && fs::file_size(dest) == size) {
            std::ifstream in(dest, std::ios::binary); std::array<char, 4096> b{};
            size_t off = 0; equal = true;
            while (off < size) {
                size_t count = std::min(b.size(), size - off);
                if (!in.read(b.data(), std::streamsize(count)) || std::memcmp(b.data(), data + off, count)) { equal = false; break; }
                off += count;
            }
        }
        if (!equal) {
            static std::atomic<unsigned> sequence{0};
#ifdef _WIN32
            const auto pid = GetCurrentProcessId();
#else
            const auto pid = getpid();
#endif
            fs::path temp = dest;
            temp += ".tmp." + std::to_string(pid) + "." + std::to_string(sequence++);
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(data), std::streamsize(size));
            out.close(); if (!out) throw std::runtime_error("program dump write failed");
#ifdef _WIN32
            if (!MoveFileExW(temp.c_str(), dest.c_str(), MOVEFILE_REPLACE_EXISTING))
                throw std::runtime_error("program dump rename failed");
#else
            fs::rename(temp, dest);
#endif
        }
        outputs.emplace_back(dest, size); return dest;
    }
    void record(const std::string& source, size_t offset, const char* kind,
                size_t size, const char* extent, const fs::path& dest) {
        report << "{\"source\":" << quote(source) << ",\"offset\":" << offset
               << ",\"kind\":" << quote(kind) << ",\"size\":" << size
               << ",\"extent\":" << quote(extent) << ",\"path\":" << quote(dest.generic_string()) << "}\n";
    }
    void scan(const byte* p, size_t n, const std::string& source, unsigned depth = 0) {
        for (size_t off = 0; off < n; ++off) {
            const byte* b = p + off; const size_t left = n - off;
            if (spu && b[0] == 0x7f) {
                if (size_t size = spu_elf_extent(b, left)) {
                    record(source, off, "spu-elf", size, "elf-tables", save("spu_jobs/boot", "spu_", ".elf", b, size));
                    ++spu_count; off += size - 1; continue;
                }
            }
            if (shaders && left >= 32 && b[0] == 0 && b[1] == 0) {
                uint32_t profile = be32(b), revision = be32(b + 4);
                if ((profile == 7001 || profile == 6151 || profile == 7003 || profile == 7004) &&
                    (revision == 6 || revision == 7)) {
                    size_t total = be32(b + 8), params = be32(b + 12), paramoff = be32(b + 16);
                    size_t program = be32(b + 20), size = be32(b + 24), ucode = be32(b + 28);
                    size_t header = revision == 7 ? 36 : 32;
                    if (total >= header && total <= left && size && size % 16 == 0 &&
                        ucode >= header && ucode <= total && size <= total - ucode &&
                        program >= header && program <= total && total - program >= 16 &&
                        (!params || (paramoff >= header && paramoff <= total && params <= (total - paramoff) / 48))) {
                        bool vp = profile == 7001 || profile == 7003;
                        auto dest = save("shaders/rsx", vp ? "vp_" : "fp_", ".cg", b, total);
                        save("shaders/rsx", vp ? "vp_" : "fp_", ".ucode", b + ucode, size);
                        record(source, off, vp ? "rsx-vp" : "rsx-fp", total, "cg-header", dest);
                        ++shader_count; off += total - 1; continue;
                    }
                }
            }
            if (shaders && b[0] == 'C') {
                if (size_t total = cgb_extent(b, left)) {
                    bool vp = b[10] == 0;
                    auto dest = save("shaders/rsx", vp ? "vp_" : "fp_", ".cgb", b, total);
                    save("shaders/rsx", vp ? "vp_" : "fp_", ".ucode", b + 32, be16(b + 8));
                    record(source, off, vp ? "rsx-vp" : "rsx-fp", total, "cgb-blocks", dest);
                    ++shader_count; off += total - 1; continue;
                }
            }
            if (spu && (b[0] == 0x42 || b[0] == 0x43) && guid(b, left)) {
                size_t bound = std::min(left, ls_size);
                for (size_t next = 16; next + 16 <= bound; next += 4)
                    if (guid(b + next, bound - next)) { bound = next; break; }
                size_t size = bound; const char* extent = "unknown-bounded-region";
                if (left >= 64 && !std::memcmp(b + 48, "JOBCRT Ver14", 12)) {
                    uint64_t candidate = uint64_t(be32(b + 60)) + 64;
                    if (candidate >= 128 && candidate <= bound && candidate % 16 == 0) {
                        size = size_t(candidate); extent = "jobcrt-header-candidate";
                    }
                }
                record(source, off, "raw-spurs-candidate", size, extent,
                    save("spu_jobs/boot", "region_", ".bin", b, size));
                ++regions; off += 15; continue;
            }
            if (left >= 16 && b[0] == 'C' && !std::memcmp(b, "CRILAYLA", 8)) {
                try {
                    auto data = depth < 4 ? unpack_crilayla(b, left) : std::vector<byte>{};
                    if (data.empty()) throw std::runtime_error("unsupported size or nesting");
                    ++compressed;
                    scan(data.data(), data.size(), source + "!CRILAYLA@" + std::to_string(off), depth + 1);
                    off += size_t(le32(b + 12)) + 16 + 256 - 1;
                } catch (const std::exception& e) {
                    ++rejected;
                    report << "{\"source\":" << quote(source) << ",\"offset\":" << off
                           << ",\"kind\":\"compressed-unreadable\",\"reason\":" << quote(e.what()) << "}\n";
                }
            }
        }
    }
};

bool beneath(const fs::path& path, const fs::path& root) {
    auto p = path.begin();
    for (auto r = root.begin(); r != root.end(); ++r, ++p)
        if (p == path.end() || *p != *r) return false;
    return true;
}
} // namespace

extern "C" void ps3_boot_collect_programs(const char* game_root, const char* image_path,
                                          const char* executable_path) {
    if (disabled("PS3RECOMP_BOOT_DUMP")) return;
    try {
        Collector c; c.root = fs::weakly_canonical(cache_root());
        if (!c.spu && !c.shaders) return;
        std::set<fs::path> inputs;
        uint64_t errors = 0, reused = 0, scanned = 0;
        std::error_code ec;
        auto add = [&](const fs::path& p) {
            auto canonical = fs::weakly_canonical(p, ec);
            if (!ec && fs::is_regular_file(canonical, ec) && !beneath(canonical, c.root)) inputs.insert(canonical);
        };
#ifdef _WIN32
        std::vector<wchar_t> module(32768);
        const DWORD count = GetModuleFileNameW(nullptr, module.data(), DWORD(module.size()));
        if (count && count < module.size()) add(fs::path(module.data()));
        else if (executable_path && *executable_path) add(executable_path);
#else
        if (executable_path && *executable_path) add(executable_path);
#endif
        if (image_path && *image_path) add(image_path);
        fs::path game = fs::weakly_canonical(game_root && *game_root ? game_root : ".");
        const std::string title = game.generic_string();
        fs::path index = c.root / "boot" / hex(hash(reinterpret_cast<const byte*>(title.data()), title.size()));
        fs::create_directories(index);
        std::ofstream failures(index / "errors.jsonl", std::ios::trunc);
        fs::recursive_directory_iterator it(game, fs::directory_options::skip_permission_denied, ec), end;
        if (ec) { ++errors; failures << "{\"source\":" << quote(title) << ",\"error\":" << quote(ec.message()) << "}\n"; }
        while (it != end) {
            auto p = it->path();
            bool link = it->is_symlink(ec);
#ifdef _WIN32
            DWORD attr = GetFileAttributesW(p.c_str());
            link = link || (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_REPARSE_POINT));
#endif
            if (link || beneath(fs::weakly_canonical(p, ec), c.root)) it.disable_recursion_pending();
            else if (it->is_regular_file(ec)) add(p);
            it.increment(ec);
            if (ec) { ++errors; failures << "{\"source\":" << quote(p.generic_string()) << ",\"error\":" << quote(ec.message()) << "}\n"; ec.clear(); }
        }
        std::fprintf(stderr, "[boot-cache] scanning %zu files; outputs: %s\n", inputs.size(), c.root.string().c_str());
        for (const auto& p : inputs) {
            try {
                const std::string name = p.generic_string();
                const uintmax_t size = fs::file_size(p);
                const auto time = fs::last_write_time(p).time_since_epoch().count();
                const std::string key = hex(hash(reinterpret_cast<const byte*>(name.data()), name.size()));
                fs::path stamp = index / (key + ".stamp"), report = index / (key + ".jsonl");
                std::ostringstream identity;
                identity << scan_version << ' ' << size << ' ' << time << ' ' << c.spu << ' ' << c.shaders;
                std::ifstream previous(stamp); std::string line;
                bool hit = bool(std::getline(previous, line)) && line == identity.str() && fs::exists(report);
                bool complete = false;
                while (hit && std::getline(previous, line)) {
                    size_t split = line.find(' ');
                    if (split == std::string::npos) { hit = false; break; }
                    if (line.substr(0, split) == "complete") {
                        uintmax_t report_size = 0;
                        std::istringstream number(line.substr(split + 1));
                        complete = bool(number >> report_size) &&
                            number.peek() == std::char_traits<char>::eof() &&
                            fs::file_size(report, ec) == report_size && !ec;
                        if (std::getline(previous, line)) complete = false;
                        break;
                    }
                    uintmax_t expected = 0;
                    std::istringstream number(line.substr(0, split));
                    if (!(number >> expected) || number.peek() != std::char_traits<char>::eof()) { hit = false; break; }
                    fs::path output = fs::weakly_canonical(c.root / fs::path(line.substr(split + 1)), ec);
                    hit = !ec && beneath(output, c.root) && fs::is_regular_file(output, ec) &&
                          fs::file_size(output, ec) == expected && !ec;
                }
                if (hit && complete) { ++reused; continue; }
                previous.close();
                fs::remove(stamp, ec); // An interrupted rescan must not reuse the old inventory.
                std::fprintf(stderr, "[boot-cache] file %llu/%zu: %s (%llu bytes)\n",
                    (unsigned long long)(scanned + reused + errors + 1), inputs.size(),
                    name.c_str(), (unsigned long long)size);
                c.outputs.clear();
                c.report.open(report, std::ios::trunc);
                if (!c.report) throw std::runtime_error("cannot open inventory");
                MappedFile input(p);
                if (size && (!input.data || input.size != size)) throw std::runtime_error("cannot map source");
                c.scan(input.data, input.size, name);
                c.report.close(); if (!c.report) throw std::runtime_error("cannot write inventory");
                if (fs::file_size(p) != size || fs::last_write_time(p).time_since_epoch().count() != time)
                    throw std::runtime_error("source changed during scan");
                fs::path stamp_temp = stamp; stamp_temp += ".tmp";
                std::ofstream saved(stamp_temp, std::ios::trunc);
                saved << identity.str() << '\n';
                for (const auto& output : c.outputs)
                    saved << output.second << ' ' << output.first.lexically_relative(c.root).generic_string() << '\n';
                saved << "complete " << fs::file_size(report) << '\n';
                saved.close(); if (!saved) throw std::runtime_error("cannot write scan stamp");
                fs::rename(stamp_temp, stamp);
                ++scanned;
            } catch (const std::exception& e) {
                c.report.close(); c.report.clear(); ++errors;
                failures << "{\"source\":" << quote(p.generic_string()) << ",\"error\":" << quote(e.what()) << "}\n";
            }
        }
        std::ofstream summary(index / "summary.json", std::ios::trunc);
        summary << "{\"version\":" << scan_version << ",\"files_scanned\":" << scanned
                << ",\"files_reused\":" << reused << ",\"errors\":" << errors
                << ",\"new_shader_instances\":" << c.shader_count << ",\"new_spu_elf_instances\":" << c.spu_count
                << ",\"new_raw_candidates\":" << c.regions << ",\"new_members_decompressed\":" << c.compressed
                << ",\"new_members_rejected\":" << c.rejected << ",\"complete_game_coverage\":false}\n";
        std::fprintf(stderr, "[boot-cache] scanned=%llu reused=%llu shaders=%llu SPU-ELFs=%llu raw-candidates=%llu errors=%llu compressed-rejected=%llu\n",
            (unsigned long long)scanned, (unsigned long long)reused, (unsigned long long)c.shader_count,
            (unsigned long long)c.spu_count, (unsigned long long)c.regions, (unsigned long long)errors,
            (unsigned long long)c.rejected);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[boot-cache] discovery failed: %s\n", e.what());
    }
}
