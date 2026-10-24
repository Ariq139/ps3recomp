#ifndef PS3RECOMP_BOOT_PROGRAM_DUMP_H
#define PS3RECOMP_BOOT_PROGRAM_DUMP_H
#ifdef __cplusplus
extern "C" {
#endif
/* Collect disk-resident programs before guest execution. No programs are run
 * or registered by this scan. See docs/BOOT_PROGRAM_CACHE.md for coverage. */
void ps3_boot_collect_programs(const char* game_root, const char* image_path,
                               const char* executable_path);
#ifdef __cplusplus
}
#endif
#endif
