#pragma once

#include <stddef.h>

/* Populate <game root>/event from USRDIR/data.cpk when no usable loose EVT
 * files are present. Existing nonempty loose EVT files always take priority. */
extern "C" bool ps3recomp_materialize_cpk_event_folder(const char* game_root);

/* Resolve June's concatenated /app_home/event<child> paths against the loose
 * event tree, returning false when the path cannot be matched safely. */
extern "C" int ps3recomp_resolve_event_editor_path(const char* game_root,
                                                   const char* guest_path,
                                                   char* host_path,
                                                   size_t host_path_capacity);

extern "C" int ps3recomp_canonicalize_cpk_event_path(const char* game_root,
                                                      const char* guest_path,
                                                      char* canonical_path,
                                                      size_t canonical_path_capacity);
