#include "cpk_event_fallback.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <strings.h>
#include <sys/types.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kMaxTableSize = 64ull * 1024 * 1024;
constexpr uint64_t kMaxExtractedFileSize = 512ull * 1024 * 1024;
constexpr size_t kUtfBaseOffset = 8;
constexpr size_t kUtfColumnOffset = 0x20;

int seek_file(FILE* file, int64_t offset, int origin)
{
#ifdef _WIN32
    return _fseeki64(file, offset, origin);
#else
    return fseeko(file, static_cast<off_t>(offset), origin);
#endif
}

int64_t tell_file(FILE* file)
{
#ifdef _WIN32
    return static_cast<int64_t>(_ftelli64(file));
#else
    return static_cast<int64_t>(ftello(file));
#endif
}

int compare_prefix_case_insensitive(const char* text, const char* prefix, size_t length)
{
#ifdef _WIN32
    return _strnicmp(text, prefix, length);
#else
    return strncasecmp(text, prefix, length);
#endif
}

uint16_t be16(const uint8_t* p)
{
    return (uint16_t(p[0]) << 8) | uint16_t(p[1]);
}

uint32_t be32(const uint8_t* p)
{
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

uint64_t be64(const uint8_t* p)
{
    return (uint64_t(be32(p)) << 32) | be32(p + 4);
}

uint32_t le32(const uint8_t* p)
{
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

size_t value_size(uint8_t type)
{
    switch (type & 0x0f) {
    case 0: case 1: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 8: case 10: return 4;
    case 6: case 7: case 9: case 11: return 8;
    case 12: return 16;
    default: return 0;
    }
}

struct UtfColumn {
    std::string name;
    uint8_t flags = 0;
    size_t size = 0;
    std::vector<uint8_t> default_value;
};

struct UtfTable {
    std::vector<uint8_t> bytes;
    size_t rows_offset = 0;
    size_t strings_offset = 0;
    uint16_t row_size = 0;
    uint32_t row_count = 0;
    std::vector<UtfColumn> columns;

    bool string_at(uint32_t offset, std::string& result) const
    {
        if (strings_offset >= bytes.size() || offset >= bytes.size() - strings_offset)
            return false;
        const char* begin = reinterpret_cast<const char*>(bytes.data() + strings_offset + offset);
        const size_t remaining = bytes.size() - strings_offset - offset;
        const void* end = std::memchr(begin, 0, remaining);
        if (!end) return false;
        result.assign(begin, static_cast<const char*>(end) - begin);
        return true;
    }

    bool row_value(uint32_t row, size_t column_index, const uint8_t*& value) const
    {
        if (row >= row_count || column_index >= columns.size()) return false;
        const UtfColumn& target = columns[column_index];
        if (!(target.flags & 0x40)) {
            if (!(target.flags & 0x20) || target.default_value.size() != target.size)
                return false;
            value = target.default_value.data();
            return true;
        }
        uint64_t row_start = uint64_t(rows_offset) + uint64_t(row) * row_size;
        if (row_start > bytes.size() || row_size > bytes.size() - row_start) return false;
        size_t cursor = static_cast<size_t>(row_start);
        for (size_t i = 0; i < columns.size(); ++i) {
            const UtfColumn& column = columns[i];
            if (column.flags & 0x40) {
                if (cursor > bytes.size() || column.size > bytes.size() - cursor) return false;
                if (i == column_index) {
                    value = bytes.data() + cursor;
                    return true;
                }
                cursor += column.size;
            }
        }
        return false;
    }

    int find_column(const char* name) const
    {
        for (size_t i = 0; i < columns.size(); ++i)
            if (columns[i].name == name) return static_cast<int>(i);
        return -1;
    }

    bool row_string(uint32_t row, int column_index, std::string& result) const
    {
        if (column_index < 0) return false;
        const uint8_t* value = nullptr;
        if (!row_value(row, static_cast<size_t>(column_index), value) ||
            columns[column_index].size != 4)
            return false;
        return string_at(be32(value), result);
    }

    bool row_number(uint32_t row, int column_index, uint64_t& result) const
    {
        if (column_index < 0) return false;
        const uint8_t* value = nullptr;
        if (!row_value(row, static_cast<size_t>(column_index), value)) return false;
        switch (columns[column_index].flags & 0x0f) {
        case 0: result = value[0]; return true;
        case 1: result = static_cast<uint64_t>(static_cast<int8_t>(value[0])); return true;
        case 2: result = be16(value); return true;
        case 3: result = static_cast<uint64_t>(static_cast<int16_t>(be16(value))); return true;
        case 4: result = be32(value); return true;
        case 5: result = static_cast<uint64_t>(static_cast<int32_t>(be32(value))); return true;
        case 6: result = be64(value); return true;
        case 7: result = be64(value); return true;
        default: return false;
        }
    }
};

bool parse_utf(std::vector<uint8_t> bytes, UtfTable& table)
{
    if (bytes.size() < kUtfColumnOffset || std::memcmp(bytes.data(), "@UTF", 4) != 0)
        return false;
    const uint64_t rows_offset = kUtfBaseOffset + be16(bytes.data() + 0x0a);
    const uint64_t strings_offset = kUtfBaseOffset + be32(bytes.data() + 0x0c);
    const uint16_t column_count = be16(bytes.data() + 0x18);
    const uint16_t row_size = be16(bytes.data() + 0x1a);
    const uint32_t row_count = be32(bytes.data() + 0x1c);
    if (!column_count || column_count > 512 || !row_size || !row_count || row_count > 1000000 ||
        rows_offset > bytes.size() || strings_offset > bytes.size() ||
        uint64_t(rows_offset) + uint64_t(row_size) * row_count > bytes.size())
        return false;

    table.bytes = std::move(bytes);
    table.rows_offset = static_cast<size_t>(rows_offset);
    table.strings_offset = static_cast<size_t>(strings_offset);
    table.row_size = row_size;
    table.row_count = row_count;
    table.columns.clear();
    table.columns.reserve(column_count);

    size_t cursor = kUtfColumnOffset;
    for (uint16_t i = 0; i < column_count; ++i) {
        if (cursor >= table.bytes.size()) return false;
        UtfColumn column;
        column.flags = table.bytes[cursor++];
        column.size = value_size(column.flags);
        if (!column.size) return false;
        if (column.flags & 0x10) {
            if (cursor + 4 > table.bytes.size()) return false;
            std::string name;
            if (!table.string_at(be32(table.bytes.data() + cursor), name)) return false;
            column.name = std::move(name);
            cursor += 4;
        }
        if (column.flags & 0x20) {
            if (cursor > table.bytes.size() || column.size > table.bytes.size() - cursor)
                return false;
            column.default_value.assign(table.bytes.begin() + cursor,
                                        table.bytes.begin() + cursor + column.size);
            cursor += column.size;
        }
        table.columns.push_back(std::move(column));
    }
    return true;
}

bool read_table(FILE* archive, uint64_t archive_size, uint64_t offset, UtfTable& table)
{
    uint8_t container[16];
    if (offset > archive_size || sizeof(container) > archive_size - offset ||
        seek_file(archive, static_cast<int64_t>(offset), SEEK_SET) != 0 ||
        std::fread(container, 1, sizeof(container), archive) != sizeof(container))
        return false;
    const uint32_t size = le32(container + 8);
    if (!size || size > kMaxTableSize || uint64_t(size) > archive_size - offset - sizeof(container))
        return false;
    std::vector<uint8_t> bytes(size);
    if (std::fread(bytes.data(), 1, bytes.size(), archive) != bytes.size()) return false;
    return parse_utf(std::move(bytes), table);
}

bool event_archive_entries(FILE* archive, uint64_t archive_size,
                           std::vector<struct CpkEntry>& entries);

struct CpkEntry {
    std::string path;
    uint64_t offset = 0;
    uint64_t stored_size = 0;
};

bool event_archive_entries(FILE* archive, uint64_t archive_size,
                           std::vector<CpkEntry>& entries)
{
    UtfTable header;
    if (!read_table(archive, archive_size, 0, header)) return false;
    uint64_t toc_offset = 0, content_offset = 0;
    if (!header.row_number(0, header.find_column("TocOffset"), toc_offset) ||
        !header.row_number(0, header.find_column("ContentOffset"), content_offset))
        return false;
    if (toc_offset < content_offset) content_offset = toc_offset;

    UtfTable toc;
    if (!read_table(archive, archive_size, toc_offset, toc)) return false;
    const int dir_col = toc.find_column("DirName");
    const int name_col = toc.find_column("FileName");
    const int size_col = toc.find_column("FileSize");
    const int offset_col = toc.find_column("FileOffset");
    if (dir_col < 0 || name_col < 0 || size_col < 0 || offset_col < 0) return false;

    for (uint32_t row = 0; row < toc.row_count; ++row) {
        std::string directory, name;
        uint64_t stored_size = 0, relative_offset = 0;
        if (!toc.row_string(row, dir_col, directory) || !toc.row_string(row, name_col, name) ||
            !toc.row_number(row, size_col, stored_size) ||
            !toc.row_number(row, offset_col, relative_offset))
            continue;
        std::replace(directory.begin(), directory.end(), '\\', '/');
        while (!directory.empty() && directory.front() == '/') directory.erase(directory.begin());
        while (!directory.empty() && directory.back() == '/') directory.pop_back();
        if (directory.size() < 5 ||
            compare_prefix_case_insensitive(directory.c_str(), "event", 5) != 0 ||
            (directory.size() > 5 && directory[5] != '/'))
            continue;
        if (name.empty() || name == "." || name == ".." ||
            name.find_first_of("/\\:") != std::string::npos ||
            directory.find("..") != std::string::npos)
            continue;
        if (content_offset > std::numeric_limits<uint64_t>::max() - relative_offset) continue;
        const uint64_t file_offset = content_offset + relative_offset;
        if (file_offset > archive_size || stored_size > archive_size - file_offset) continue;
        entries.push_back({directory + "/" + name, file_offset, stored_size});
    }
    return !entries.empty();
}

bool read_exact_at(FILE* file, uint64_t offset, void* destination, size_t size)
{
    if (seek_file(file, static_cast<int64_t>(offset), SEEK_SET) != 0) return false;
    return std::fread(destination, 1, size, file) == size;
}

bool decompress_crilayla(const std::vector<uint8_t>& input, std::vector<uint8_t>& output)
{
    if (input.size() < 16 || std::memcmp(input.data(), "CRILAYLA", 8) != 0) return false;
    const uint32_t body_size = le32(input.data() + 8);
    const uint32_t header_offset = le32(input.data() + 12);
    const uint64_t header_start = uint64_t(header_offset) + 16;
    if (!body_size || uint64_t(body_size) + 0x100 > kMaxExtractedFileSize ||
        header_start > input.size() || 0x100 > input.size() - header_start || header_start < 16)
        return false;

    output.assign(size_t(body_size) + 0x100, 0);
    std::memcpy(output.data(), input.data() + header_start, 0x100);
    size_t cursor = static_cast<size_t>(header_start);
    unsigned bits_left = 0;
    const size_t stream_begin = 16;
    auto read_bits = [&](unsigned count, uint32_t& result) -> bool {
        result = 0;
        for (unsigned i = 0; i < count; ++i) {
            if (bits_left == 0) {
                if (cursor <= stream_begin) return false;
                --cursor;
                bits_left = 8;
            }
            result = (result << 1) | ((input[cursor] >> (bits_left - 1)) & 1u);
            --bits_left;
        }
        return true;
    };

    size_t write = output.size() - 1;
    while (write >= 0x100) {
        uint32_t compressed = 0;
        if (!read_bits(1, compressed)) return false;
        if (compressed) {
            uint32_t encoded_offset = 0, part = 0;
            if (!read_bits(13, encoded_offset) || !read_bits(2, part)) return false;
            const uint64_t distance = uint64_t(encoded_offset) + 3;
            uint64_t length = 3 + part;
            if (part == 3) {
                if (!read_bits(3, part)) return false;
                length += part;
                if (part == 7) {
                    if (!read_bits(5, part)) return false;
                    length += part;
                    if (part == 31) {
                        do {
                            if (!read_bits(8, part)) return false;
                            length += part;
                        } while (part == 255);
                    }
                }
            }
            if (length > write - 0x100 + 1 || write + distance >= output.size()) return false;
            for (uint64_t i = 0; i < length; ++i) {
                output[write] = output[write + static_cast<size_t>(distance)];
                --write;
            }
        } else {
            uint32_t literal = 0;
            if (!read_bits(8, literal)) return false;
            output[write--] = static_cast<uint8_t>(literal);
        }
    }
    return write == 0xff;
}

bool read_archive_file(FILE* archive, uint64_t archive_size, const CpkEntry& entry,
                       std::vector<uint8_t>& data)
{
    if (!entry.stored_size || entry.stored_size > kMaxExtractedFileSize ||
        entry.offset > archive_size || entry.stored_size > archive_size - entry.offset)
        return false;
    data.resize(static_cast<size_t>(entry.stored_size));
    if (!read_exact_at(archive, entry.offset, data.data(), data.size())) return false;

    if (data.size() >= 8 && std::memcmp(data.data(), "CRILAYLA", 8) == 0) {
        if (data.size() < 16) return false;
        const uint64_t needed = uint64_t(le32(data.data() + 12)) + 16 + 0x100;
        if (needed > kMaxExtractedFileSize || needed > archive_size - entry.offset) return false;
        if (needed > data.size()) {
            data.resize(static_cast<size_t>(needed));
            if (!read_exact_at(archive, entry.offset, data.data(), data.size())) return false;
        }
        std::vector<uint8_t> unpacked;
        if (!decompress_crilayla(data, unpacked)) return false;
        data = std::move(unpacked);
    }
    return true;
}

bool has_nonempty_evt(const fs::path& root)
{
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return false;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string extension = it->path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".evt" && it->file_size(ec) > 0) return true;
    }
    return false;
}

bool safe_event_path(const std::string& archive_path, fs::path& relative)
{
    std::string normalized = archive_path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    if (normalized.size() < 7 || compare_prefix_case_insensitive(normalized.c_str(), "event/", 6) != 0)
        return false;
    normalized.erase(0, 6);
    if (normalized.empty() || normalized.front() == '/' || normalized.find(':') != std::string::npos)
        return false;
    size_t start = 0;
    while (start <= normalized.size()) {
        const size_t end = normalized.find('/', start);
        const std::string part = normalized.substr(start, end == std::string::npos ? end : end - start);
        if (part.empty() || part == "." || part == "..") return false;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    relative = fs::path(normalized);
    return true;
}

bool same_prefix_ci(const std::string& value, const std::string& prefix)
{
    if (value.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        const unsigned char a = static_cast<unsigned char>(value[i]);
        const unsigned char b = static_cast<unsigned char>(prefix[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

bool resolve_flat_event_tail(const fs::path& directory, const std::string& tail,
                             unsigned depth, fs::path& resolved, size_t& consumed)
{
    consumed = 0;
    if (depth > 16 || tail.size() > 1024) return false;
    std::string remainder = tail;
    while (!remainder.empty() && remainder.front() == '/') remainder.erase(remainder.begin());
    if (remainder.empty()) {
        resolved = directory;
        return true;
    }
    if (remainder.find(':') != std::string::npos || remainder.find("..") != std::string::npos)
        return false;

    std::error_code ec;
    const fs::path direct = directory / fs::path(remainder);
    if (fs::exists(direct, ec) && !ec) {
        resolved = direct;
        consumed = remainder.size();
        return true;
    }

    fs::path best_path;
    size_t best_consumed = 0;
    fs::directory_iterator it(directory, fs::directory_options::skip_permission_denied, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code entry_ec;
        if (it->is_symlink(entry_ec) || !it->is_directory(entry_ec)) continue;
        const std::string name = it->path().filename().string();
        if (name.empty() || !same_prefix_ci(remainder, name)) continue;
        std::string child_tail = remainder.substr(name.size());
        if (!child_tail.empty() && child_tail.front() == '/') child_tail.erase(child_tail.begin());
        if (child_tail.empty()) {
            if (name.size() > best_consumed) {
                best_path = it->path();
                best_consumed = name.size();
            }
            continue;
        }
        fs::path child_path;
        size_t child_consumed = 0;
        if (resolve_flat_event_tail(it->path(), child_tail, depth + 1, child_path,
                                    child_consumed)) {
            const size_t total_consumed = name.size() + child_consumed;
            if (total_consumed > best_consumed) {
                best_path = std::move(child_path);
                best_consumed = total_consumed;
            }
        } else if (name.size() > best_consumed) {
            best_path = it->path();
            best_consumed = name.size();
        }
    }
    if (!best_consumed) return false;
    resolved = std::move(best_path);
    consumed = best_consumed;
    return true;
}

bool extract_events(const fs::path& archive_path, const fs::path& event_root)
{
    FILE* archive = nullptr;
#ifdef _WIN32
    if (_wfopen_s(&archive, archive_path.c_str(), L"rb") != 0) archive = nullptr;
#else
    archive = std::fopen(archive_path.string().c_str(), "rb");
#endif
    if (!archive) return false;
    struct CloseFile { FILE* f; ~CloseFile() { if (f) std::fclose(f); } } close{archive};
    if (seek_file(archive, 0, SEEK_END) != 0) return false;
    const int64_t size_signed = tell_file(archive);
    if (size_signed <= 0) return false;
    const uint64_t archive_size = static_cast<uint64_t>(size_signed);

    std::vector<CpkEntry> entries;
    if (!event_archive_entries(archive, archive_size, entries)) return false;
    std::error_code ec;
    fs::create_directories(event_root, ec);
    if (ec) return false;

    size_t written_evt = 0;
    size_t skipped = 0;
    for (const CpkEntry& entry : entries) {
        fs::path relative;
        if (!safe_event_path(entry.path, relative)) { ++skipped; continue; }
        const fs::path destination = event_root / relative;
        ec.clear();
        if (fs::is_regular_file(destination, ec) && fs::file_size(destination, ec) > 0) {
            ++skipped;
            continue;
        }
        ec.clear();
        fs::create_directories(destination.parent_path(), ec);
        if (ec) { ++skipped; continue; }

        std::vector<uint8_t> contents;
        if (!read_archive_file(archive, archive_size, entry, contents) || contents.empty()) {
            ++skipped;
            continue;
        }
        FILE* output = nullptr;
#ifdef _WIN32
        if (_wfopen_s(&output, destination.c_str(), L"wb") != 0) output = nullptr;
#else
        output = std::fopen(destination.string().c_str(), "wb");
#endif
        if (!output) { ++skipped; continue; }
        const bool wrote = std::fwrite(contents.data(), 1, contents.size(), output) == contents.size();
        const bool closed = std::fclose(output) == 0;
        if (!wrote || !closed) {
            fs::remove(destination, ec);
            ++skipped;
            continue;
        }
        std::string extension = destination.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".evt") ++written_evt;
    }
    std::fprintf(stderr, "[fs] event CPK fallback: extracted %zu EVT files from '%s' (%zu skipped)\n",
                 written_evt, archive_path.string().c_str(), skipped);
    return written_evt > 0 || has_nonempty_evt(event_root);
}

} // namespace

extern "C" bool ps3recomp_materialize_cpk_event_folder(const char* game_root)
{
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    const fs::path root = (game_root && *game_root) ? fs::path(game_root) : fs::path(".");
    const fs::path event_root = root / "event";
    if (has_nonempty_evt(event_root)) return true;

    const fs::path candidates[] = {
        root / "USRDIR" / "data.cpk",
        root / "usrdir" / "data.cpk",
        root / "data.cpk"
    };
    for (const fs::path& archive : candidates) {
        std::error_code ec;
        if (!fs::is_regular_file(archive, ec)) continue;
        if (extract_events(archive, event_root)) return true;
    }
    return false;
}

extern "C" int ps3recomp_canonicalize_cpk_event_path(const char* game_root,
                                                      const char* guest_path,
                                                      char* canonical_path,
                                                      size_t canonical_path_capacity)
{
    if (!guest_path || !canonical_path || canonical_path_capacity == 0) return 0;

    std::string flattened;
    flattened.reserve(std::strlen(guest_path));
    const char* input = guest_path;
    while (*input == '/' || *input == '\\') ++input;
    if (compare_prefix_case_insensitive(input, "app_home/", 9) == 0 ||
        compare_prefix_case_insensitive(input, "app_home\\", 9) == 0)
        input += 9;
    else if (compare_prefix_case_insensitive(input, "app_home", 8) == 0 &&
             compare_prefix_case_insensitive(input + 8, "event", 5) == 0)
        input += 8;
    for (; *input; ++input)
        if (*input != '/' && *input != '\\')
            flattened.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*input))));

    if (flattened.size() < 10 || flattened.compare(0, 5, "event") != 0)
        return 0;

    const fs::path root = (game_root && *game_root) ? fs::path(game_root) : fs::path(".");
    const fs::path candidates[] = {
        root / "USRDIR" / "data.cpk",
        root / "usrdir" / "data.cpk",
        root / "data.cpk"
    };

    for (const fs::path& archive_path : candidates) {
        std::error_code ec;
        if (!fs::is_regular_file(archive_path, ec)) continue;
        FILE* archive = nullptr;
#ifdef _WIN32
        if (_wfopen_s(&archive, archive_path.c_str(), L"rb") != 0) archive = nullptr;
#else
        archive = std::fopen(archive_path.string().c_str(), "rb");
#endif
        if (!archive) continue;
        struct CloseFile { FILE* f; ~CloseFile() { if (f) std::fclose(f); } } close{archive};
        if (seek_file(archive, 0, SEEK_END) != 0) continue;
        const int64_t size_signed = tell_file(archive);
        if (size_signed <= 0) continue;

        std::vector<CpkEntry> entries;
        if (!event_archive_entries(archive, static_cast<uint64_t>(size_signed), entries)) continue;
        for (const CpkEntry& entry : entries) {
            std::string entry_flattened;
            entry_flattened.reserve(entry.path.size());
            for (unsigned char ch : entry.path)
                if (ch != '/' && ch != '\\')
                    entry_flattened.push_back(static_cast<char>(std::tolower(ch)));
            if (entry_flattened != flattened) continue;
            if (entry.path.size() >= canonical_path_capacity) return 0;
            std::memcpy(canonical_path, entry.path.c_str(), entry.path.size() + 1);
            return 1;
        }
    }
    return 0;
}

extern "C" int ps3recomp_resolve_event_editor_path(const char* game_root,
                                                   const char* guest_path,
                                                   char* host_path,
                                                   size_t host_path_capacity)
{
    static const char prefix[] = "/app_home/event";
    if (!guest_path || !host_path || !host_path_capacity) return 0;

    const char* tail = nullptr;
    if (std::strncmp(guest_path, prefix, sizeof(prefix) - 1) == 0 &&
        guest_path[sizeof(prefix) - 1] != '\0' && guest_path[sizeof(prefix) - 1] != '/') {
        tail = guest_path + sizeof(prefix) - 1;
    } else if (std::strncmp(guest_path, "event", 5) == 0 &&
               guest_path[5] != '\0' && guest_path[5] != '/') {
        tail = guest_path + 5;
    } else {
        return 0;
    }

    if (!ps3recomp_materialize_cpk_event_folder(game_root)) return 0;
    const fs::path event_root = ((game_root && *game_root) ? fs::path(game_root) : fs::path(".")) / "event";
    fs::path resolved;
    size_t consumed = 0;
    if (!resolve_flat_event_tail(event_root, tail, 0, resolved, consumed)) return 0;

    const std::string native_path = resolved.string();
    if (native_path.size() >= host_path_capacity) return 0;
    std::memcpy(host_path, native_path.c_str(), native_path.size() + 1);
    for (char* p = host_path; *p; ++p) if (*p == '\\') *p = '/';
    return 1;
}
