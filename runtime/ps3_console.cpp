/* Host console policy for the game runner.
 *
 * Keep the guest's sys_tty_write stream separate from host diagnostics, decode
 * Japanese CP932 text to UTF-8 and keep the guest TTY stream separate from
 * host diagnostics. Host stdout/stderr otherwise pass through unchanged.
 */
#include "ps3_console.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <Windows.h>
#else
#include <cerrno>
#include <iconv.h>
#include <unistd.h>
#endif

namespace {

struct CapturedStream {
    int original_fd = -1;
    std::thread pump;
};

std::array<CapturedStream, 2> g_streams;
std::mutex g_init_mutex;
std::mutex g_tty_mutex;
bool g_initialized = false;
std::atomic<bool> g_shutting_down{false};
FILE* g_tty_file = nullptr;
std::string g_tty_path;
bool g_tty_open_failed = false;
std::string g_tty_pending;

void write_fd(int fd, const char* data, size_t length)
{
    while (length) {
#ifdef _WIN32
        const unsigned int chunk = static_cast<unsigned int>(
            (std::min)(length, static_cast<size_t>(0x7fffffff)));
        const int written = _write(fd, data, chunk);
#else
        const ssize_t written = ::write(fd, data, length);
#endif
        if (written <= 0) return;
        data += written;
        length -= static_cast<size_t>(written);
    }
}

void pump_stream(int read_fd, int original_fd)
{
    std::string line;
    std::array<char, 4096> chunk{};
    for (;;) {
#ifdef _WIN32
        const int got = _read(read_fd, chunk.data(), static_cast<unsigned int>(chunk.size()));
#else
        const ssize_t got = ::read(read_fd, chunk.data(), chunk.size());
#endif
        if (got <= 0) break;
        for (int i = 0; i < got; ++i) {
            const char c = chunk[static_cast<size_t>(i)];
            line.push_back(c);
            if (c == '\n' || c == '\r' || line.size() >= 1024u * 1024u) {
                write_fd(original_fd, line.data(), line.size());
                line.clear();
            }
        }
    }
    if (!line.empty())
        write_fd(original_fd, line.data(), line.size());
#ifdef _WIN32
    _close(read_fd);
#else
    ::close(read_fd);
#endif
}

bool redirect_stream(int fd)
{
#ifdef _WIN32
    int pipe_fds[2] = {-1, -1};
    if (_pipe(pipe_fds, 64 * 1024, _O_BINARY | _O_NOINHERIT) != 0) return false;
    const int original = _dup(fd);
    if (original < 0 || _dup2(pipe_fds[1], fd) != 0) {
        if (original >= 0) _close(original);
        _close(pipe_fds[0]);
        _close(pipe_fds[1]);
        return false;
    }
    _close(pipe_fds[1]);
#else
    int pipe_fds[2] = {-1, -1};
    if (::pipe(pipe_fds) != 0) return false;
    const int original = ::dup(fd);
    if (original < 0 || ::dup2(pipe_fds[1], fd) < 0) {
        if (original >= 0) ::close(original);
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        return false;
    }
    ::close(pipe_fds[1]);
#endif
    g_streams[static_cast<size_t>(fd - 1)].original_fd = original;
    g_streams[static_cast<size_t>(fd - 1)].pump =
        std::thread(pump_stream, pipe_fds[0], original);
    return true;
}

bool enabled_env(const char* name)
{
    const char* value = std::getenv(name);
    return value && *value && *value != '0' && std::strcmp(value, "false") != 0;
}

bool cp932_selected()
{
    const char* encoding = std::getenv("PS3_TTY_ENCODING");
    return !encoding || !*encoding || std::strcmp(encoding, "cp932") == 0 ||
        std::strcmp(encoding, "sjis") == 0 || std::strcmp(encoding, "shift-jis") == 0;
}

std::string decode_cp932(const char* bytes, size_t length)
{
    std::string utf8;
    utf8.reserve(length * 2);
#ifdef _WIN32
    size_t offset = 0;
    while (offset < length) {
        const unsigned char first = static_cast<unsigned char>(bytes[offset]);
        size_t char_length = 1;
        if (((first >= 0x81 && first <= 0x9f) || (first >= 0xe0 && first <= 0xfc))) {
            if (offset + 1 >= length) {
                utf8.append("\xef\xbf\xbd", 3);
                ++offset;
                continue;
            }
            const unsigned char trail = static_cast<unsigned char>(bytes[offset + 1]);
            if ((trail >= 0x40 && trail <= 0x7e) || (trail >= 0x80 && trail <= 0xfc))
                char_length = 2;
            else {
                utf8.append("\xef\xbf\xbd", 3);
                ++offset;
                continue;
            }
        }
        wchar_t wide[4]{};
        const int wide_count = MultiByteToWideChar(932, MB_ERR_INVALID_CHARS,
            bytes + offset, static_cast<int>(char_length), wide, 4);
        if (wide_count <= 0) {
            utf8.append("\xef\xbf\xbd", 3);
            ++offset;
            continue;
        }
        char encoded[16]{};
        const int encoded_count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
            wide, wide_count, encoded, static_cast<int>(sizeof(encoded)), nullptr, nullptr);
        if (encoded_count > 0) utf8.append(encoded, static_cast<size_t>(encoded_count));
        else utf8.append("\xef\xbf\xbd", 3);
        offset += char_length;
    }
#else
    iconv_t converter = iconv_open("UTF-8", "CP932");
    if (converter == (iconv_t)-1) return std::string(bytes, length);
    size_t offset = 0;
    while (offset < length) {
        const unsigned char first = static_cast<unsigned char>(bytes[offset]);
        size_t char_length = 1;
        if ((first >= 0x81 && first <= 0x9f) || (first >= 0xe0 && first <= 0xfc)) {
            if (offset + 1 < length) char_length = 2;
            else {
                utf8.append("\xef\xbf\xbd", 3);
                ++offset;
                continue;
            }
        }
        char* input = const_cast<char*>(bytes + offset);
        size_t input_left = char_length;
        char output[16]{};
        char* out = output;
        size_t output_left = sizeof(output);
        const size_t rc = iconv(converter, &input, &input_left, &out, &output_left);
        if (rc == static_cast<size_t>(-1) || input_left != 0) {
            utf8.append("\xef\xbf\xbd", 3);
            ++offset;
            iconv(converter, nullptr, nullptr, nullptr, nullptr);
            continue;
        }
        utf8.append(output, static_cast<size_t>(out - output));
        offset += char_length;
    }
    iconv_close(converter);
#endif
    return utf8;
}

std::string tty_to_utf8(const char* bytes, size_t length)
{
    if (cp932_selected())
        return decode_cp932(bytes, length);
    return std::string(bytes, length); /* utf8 or raw */
}

} // namespace

extern "C" void ps3_console_init(void)
{
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialized || g_shutting_down.load()) return;
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    redirect_stream(1);
    redirect_stream(2);
    g_initialized = true;
    std::atexit(ps3_console_shutdown);
}

extern "C" void ps3_console_shutdown(void)
{
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (!g_initialized) return;
    g_shutting_down.store(true);
    std::fflush(stdout);
    std::fflush(stderr);
    for (int fd = 1; fd <= 2; ++fd) {
        CapturedStream& stream = g_streams[static_cast<size_t>(fd - 1)];
        if (stream.original_fd < 0) continue;
#ifdef _WIN32
        _dup2(stream.original_fd, fd);
#else
        ::dup2(stream.original_fd, fd);
#endif
    }
    for (int fd = 1; fd <= 2; ++fd) {
        CapturedStream& stream = g_streams[static_cast<size_t>(fd - 1)];
        if (stream.pump.joinable()) stream.pump.join();
        if (stream.original_fd >= 0) {
#ifdef _WIN32
            _close(stream.original_fd);
#else
            ::close(stream.original_fd);
#endif
            stream.original_fd = -1;
        }
    }
    {
        std::lock_guard<std::mutex> tty_lock(g_tty_mutex);
        if (!g_tty_pending.empty()) {
            const std::string replacement("\xef\xbf\xbd", 3);
            if (g_tty_file) std::fwrite(replacement.data(), 1, replacement.size(), g_tty_file);
            g_tty_pending.clear();
        }
        if (g_tty_file) {
            std::fclose(g_tty_file);
            g_tty_file = nullptr;
        }
    }
    g_initialized = false;
}

extern "C" void ps3_console_write_tty(uint32_t channel, const void* bytes, size_t length)
{
    (void)channel;
    if (!bytes || !length) return;

    const char* setting = std::getenv("PS3_TTY_LOG");
    const bool log_disabled = setting && (!*setting || *setting == '0' || std::strcmp(setting, "off") == 0);
    const bool echo = enabled_env("PS3_TTY_STDOUT");
    if (log_disabled && !echo) return;

    ps3_console_init();
    std::lock_guard<std::mutex> lock(g_tty_mutex);
    if (g_shutting_down.load()) return;

    std::string raw;
    if (cp932_selected()) {
        raw.swap(g_tty_pending);
        raw.append(static_cast<const char*>(bytes), length);
        if (!raw.empty()) {
            const unsigned char last = static_cast<unsigned char>(raw.back());
            if ((last >= 0x81 && last <= 0x9f) || (last >= 0xe0 && last <= 0xfc)) {
                g_tty_pending.assign(1, raw.back());
                raw.pop_back();
            }
        }
    } else {
        raw.assign(static_cast<const char*>(bytes), length);
    }
    const std::string text = tty_to_utf8(raw.data(), raw.size());
    if (text.empty()) return;

    if (!log_disabled) {
        if (!g_tty_file) {
            g_tty_path = setting && *setting ? setting : "ps3_tty.log";
#ifdef _WIN32
            g_tty_file = _fsopen(g_tty_path.c_str(), "ab", _SH_DENYNO);
#else
            g_tty_file = std::fopen(g_tty_path.c_str(), "ab");
#endif
            if (g_tty_file) std::setvbuf(g_tty_file, nullptr, _IOFBF, 64 * 1024);
            else if (!g_tty_open_failed) {
                g_tty_open_failed = true;
                std::fprintf(stderr, "[tty] could not open log file '%s'\n", g_tty_path.c_str());
            }
        }
        if (g_tty_file) {
            std::fwrite(text.data(), 1, text.size(), g_tty_file);
            if (text.find('\n') != std::string::npos) std::fflush(g_tty_file);
        }
    }

    if (echo) {
        const int output_fd = g_streams[0].original_fd >= 0 ? g_streams[0].original_fd : 1;
        write_fd(output_fd, text.data(), text.size());
    }
}
