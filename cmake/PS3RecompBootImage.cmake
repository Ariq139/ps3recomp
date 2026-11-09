# Configure a port runner to use its PPU image without a launch-time file arg.
# Windows builds embed the image as RCDATA; other platforms keep the configured
# path as the runner default because their executable resource formats differ.
function(ps3recomp_configure_boot_image target source_dir)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "" "" "CANDIDATES")

    set(PORT_PPU_ELF "" CACHE FILEPATH
        "PPU ELF used to initialize guest memory (embedded in Windows builds)")

    set(boot_image "${PORT_PPU_ELF}")
    if(NOT boot_image)
        foreach(candidate IN LISTS ARG_CANDIDATES)
            if(EXISTS "${candidate}")
                set(boot_image "${candidate}")
                break()
            endif()
        endforeach()
    endif()

    if(NOT boot_image)
        message(STATUS "No default PPU image found; pass an ELF path at launch")
        return()
    endif()

    get_filename_component(boot_image "${boot_image}" ABSOLUTE
                           BASE_DIR "${source_dir}")
    if(NOT EXISTS "${boot_image}")
        message(WARNING "Configured PPU image does not exist: ${boot_image}")
        return()
    endif()

    if(WIN32)
        file(TO_NATIVE_PATH "${boot_image}" boot_image_native)
        string(REPLACE "\\" "\\\\" boot_image_rc "${boot_image_native}")
        set(PS3RECOMP_BOOT_IMAGE_RC_PATH "${boot_image_rc}")
        set(resource_file "${CMAKE_CURRENT_BINARY_DIR}/ps3recomp_boot_image.rc")
        configure_file(
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ps3recomp_boot_image.rc.in"
            "${resource_file}" @ONLY)
        set_source_files_properties("${resource_file}" PROPERTIES
            OBJECT_DEPENDS "${boot_image}")
        target_sources(${target} PRIVATE "${resource_file}")
        set_property(SOURCE "${source_dir}/main.cpp" APPEND PROPERTY
            COMPILE_DEFINITIONS PS3RECOMP_EMBED_PPU_ELF=1)
        message(STATUS "Embedded PPU boot image: ${boot_image}")
    else()
        set_property(SOURCE "${source_dir}/main.cpp" APPEND PROPERTY
            COMPILE_DEFINITIONS "PS3RECOMP_DEFAULT_PPU_ELF=\"${boot_image}\"")
        message(STATUS "Default PPU boot image: ${boot_image}")
    endif()
endfunction()
