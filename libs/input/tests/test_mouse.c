/* Standalone Windows regression test: link this file with cellMouse.c and
 * include/. The VM guards verify the PS3's six-byte reports, not host padding. */
#include "../cellMouse.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

static unsigned char guest[1024];
#define DATA_EA 0x1000u

void vm_write8(unsigned long long ea, unsigned char value)
{
    assert(ea >= DATA_EA && ea - DATA_EA < sizeof(guest));
    guest[ea - DATA_EA] = value;
}
void vm_write16(unsigned long long ea, unsigned short value)
{
    vm_write8(ea, (unsigned char)(value >> 8));
    vm_write8(ea + 1, (unsigned char)value);
}
void vm_write32(unsigned long long ea, unsigned int value)
{
    vm_write16(ea, (unsigned short)(value >> 16));
    vm_write16(ea + 2, (unsigned short)value);
}
static unsigned list(int* x, int* y)
{
    memset(guest, 0xA5, sizeof(guest));
    assert(cellMouseGetDataList(0, (CellMouseDataList*)(uintptr_t)DATA_EA) == 0);
    unsigned count = (unsigned)guest[0] << 24 | (unsigned)guest[1] << 16 |
                     (unsigned)guest[2] << 8 | guest[3];
    assert(count <= 8);
    *x = *y = 0;
    for (unsigned i = 0; i < count; i++) {
        assert(guest[4 + i * 6] == 1);
        *x += (int8_t)guest[6 + i * 6];
        *y += (int8_t)guest[7 + i * 6];
    }
    for (unsigned i = 4 + count * 6; i < sizeof(guest); i++)
        assert(guest[i] == 0xA5);
    return count;
}
int main(void)
{
    assert(sizeof(CellMouseData) == 6 && sizeof(CellMouseDataList) == 52);
    assert(!cellMouse_isInitialized());
    assert(cellMouseInit(1) == 0 && cellMouse_isInitialized());
    cellMouse_moveEvent(0, 500, -400);
    int x, y;
    assert(list(&x, &y) == 4 && x == 500 && y == -400);
    assert(list(&x, &y) == 0);

    cellMouse_moveEvent(0, 500, 0);
    cellMouse_wheelEvent(0, 1);
    cellMouse_tiltEvent(0, -1);
    assert(list(&x, &y) == 4 && x == 500);
    assert(guest[4 + 3 * 6 + 4] == 1);
    assert((int8_t)guest[4 + 3 * 6 + 5] == -1);

    cellMouse_moveEvent(0, 2048, -2048);
    int total_x = 0, total_y = 0;
    for (unsigned i = 0; i < 4; i++) {
        list(&x, &y); total_x += x; total_y += y;
    }
    assert(total_x == 2048 && total_y == -2048);

    cellMouse_positionEvent(0, 1000, 650);
    cellMouse_syncPosition(0, 123, 321);
    assert(list(&x, &y) == 7 && x == 877 && y == 329);
    cellMouse_positionEvent(0, 50, 60);
    cellMouse_syncPosition(0, 1000, 650);
    assert(list(&x, &y) == 8 && x == -950 && y == -590);
    cellMouse_syncPosition(0, 50, 60);
    assert(list(&x, &y) == 0);
    /* Guest cursor reset without another Windows movement. */
    cellMouse_syncPosition(0, 0, 0);
    assert(list(&x, &y) == 1 && x == 50 && y == 60);

    cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_1, 1);
    assert(list(&x, &y) == 1 && guest[5] == CELL_MOUSE_BUTTON_1);
    cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_1, 0);
    assert(list(&x, &y) == 1 && guest[5] == 0);
    cellMouse_moveEvent(0, 300, -300);
    int sum_x = 0, sum_y = 0;
    for (unsigned i = 0; i < 3; i++) {
        memset(guest, 0xA5, sizeof(guest));
        assert(cellMouseGetData(0, (CellMouseData*)(uintptr_t)DATA_EA) == 0);
        sum_x += (int8_t)guest[2]; sum_y += (int8_t)guest[3];
        assert(guest[6] == 0xA5 && guest[7] == 0xA5);
    }
    assert(sum_x == 300 && sum_y == -300);
    assert(cellMouseEnd() == 0 && !cellMouse_isInitialized());
    puts("mouse: ABI bounds, retained motion, pointer feedback, buttons PASS");
    return 0;
}
