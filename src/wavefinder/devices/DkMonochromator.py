import asyncio
import sys
from queue import Empty, SimpleQueue

from serial import Serial, SerialException

from ..gui.utils import Cyclic


class DkMonochromator(Cyclic):
    """Interface for Spectral Products DK series Monochromator"""

    READY = 0
    BUSY = 1
    ERROR = 2
    STATES = [READY, BUSY, ERROR]

    def __init__(self, port: str) -> None:
        self.port = Serial(
            baudrate=9600,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0,  # non-blocking mode
            rtscts=True,
            write_timeout=1.0,
            dsrdtr=True,
        )

        self.q = SimpleQueue()  # command queue
        self.port.port = port
        self.comm_up = False  # 2-way communication
        self.status = DkMonochromator.BUSY  # device status
        self.serial_number = 0
        self.target_wavelength = 0.0
        self.current_wavelength = 0.0
        self.target_slit1 = 0.0
        self.current_slit1 = 0.0
        self.target_slit2 = 0.0
        self.current_slit2 = 0.0

        try:
            print(f"Connecting to monochromator on {port}... ", flush=True)
            self.port.open()
        except (SerialException, ValueError) as e:
            self.status = DkMonochromator.ERROR
            self.port.close()
            print(e)

    async def read_bytes(self, n: int = 1, timeout: float | None = 5) -> bytes:
        """Read n bytes, async

        Args:
            n: number of bytes to read
            timeout: time in seconds to wait, or None to wait forever
        """
        if timeout is None:
            timeout = sys.float_info.max
        if timeout <= 0:
            timeout = 0.1
        result = bytes()
        while n > 0:
            if timeout <= 0:
                raise SerialException("read timeout")
            b = self.port.read(n)
            if b:
                n -= len(b)
                result = result + b
            else:
                timeout -= 0.1
                await asyncio.sleep(0.1)
        return result

    async def read_status_end(self, timeout: float | None = 5):
        """Read status byte and cancel/end byte

        Raises SerialException if status is not zero or end byte is wrong

        Args:
            timeout: time in seconds to wait, or None to wait forever
        """
        s = await self.read_bytes(2, timeout)
        # NOTE: Python yields an int when taking a slice of a bytes object...
        #       so work in ints for bitwise ops.
        not_acceptable = bool(s[0] & int.from_bytes(b"\x80"))
        equal_to_present = bool(s[0] & int.from_bytes(b"\x40"))
        if not_acceptable and not equal_to_present:
            # not_acceptable and equal_to_present is ok, just does nothing
            raise SerialException("input not acceptable")
        if s[1] != 24:
            raise SerialException("bad data")

    async def establish_connection(self) -> bool:
        """Send ECHO command, look for reply

        Returns True when communication is established.
        """
        while self.port.is_open:
            try:
                # send ECHO and wait 30 seconds for reply
                self.port.write(int(27).to_bytes())
                b = await self.read_bytes(1, 30)
                if b == int(27).to_bytes():
                    print("monochromator communication established")
                    return True
            except SerialException:
                continue  # try again
        raise SerialException("serial port is closed")

    async def get_sn(self) -> int:
        """Read serial number from monochromator"""
        sn = 0
        self.port.write(int(33).to_bytes())
        ack = await self.read_bytes()
        if ack != int(33).to_bytes():
            raise SerialException("bad ack")
        # read 5 bytes and form the sn
        sn_bytes = await self.read_bytes(5)
        sn = int(sn_bytes.decode())
        # status & end bytes
        await self.read_status_end()
        return sn

    async def get_current_wavelength(self) -> float:
        """Get current wavelength in nanometers"""
        self.port.write(int(29).to_bytes())
        ack = await self.read_bytes()
        if ack != int(29).to_bytes():
            raise SerialException("bad ack")
        # read 3 bytes and form the wavelength
        wavelength = float.fromhex((await self.read_bytes(3)).hex()) / 100
        # status & end bytes
        await self.read_status_end()
        return wavelength

    async def go_to_target_wavelength(self):
        """Command monochromater to go to target_wavelength"""
        # maximum 3-byte number is 0xFFFFFF = 16777215,
        # so max wavelength is 167772.15nm
        if self.target_wavelength > 167772.15:
            self.target_wavelength = self.current_wavelength
            raise ValueError("wavelength out of range")
        self.port.write(int(16).to_bytes())
        ack = await self.read_bytes()
        if ack != int(16).to_bytes():
            raise SerialException("bad ack")
        # convert wavelength to 3 bytes and send
        b = int(round(self.target_wavelength * 100)).to_bytes(3)
        self.port.write(b)
        await self.read_status_end(timeout=30)

    async def step_up(self):
        """Move grating one step towards IR"""
        self.port.write(int(7).to_bytes())
        ack = await self.read_bytes()
        if ack != int(7).to_bytes():
            raise SerialException("bad ack")
        await self.read_status_end()

    async def step_down(self):
        """Move grating one step towards UV"""
        self.port.write(int(1).to_bytes())
        ack = await self.read_bytes()
        if ack != int(1).to_bytes():
            raise SerialException("bad ack")
        await self.read_status_end()

    async def wait_for_wavelength_and_slits(self):
        """Wait until current wavelength and slits are their targets"""
        while (
            self.current_wavelength != self.target_wavelength
            or self.current_slit1 != self.target_slit1
            or self.current_slit2 != self.target_slit2
        ):
            await asyncio.sleep(0.1)

    async def get_current_slits(self):
        """Get current slit widths in microns"""
        self.port.write(int(30).to_bytes())
        ack = await self.read_bytes()
        if ack != int(30).to_bytes():
            raise SerialException("bad ack")
        # read 3 bytes and form the wavelength
        s1 = float.fromhex((await self.read_bytes(2)).hex())
        s2 = float.fromhex((await self.read_bytes(2)).hex())
        # status & end bytes
        await self.read_status_end()
        return (s1, s2)

    async def go_to_slit1(self):
        """Command monochromater to go to slit1 position"""
        # must be between 10 and 3000 microns
        if self.target_slit1 > 3000 or self.target_slit1 < 10:
            self.target_slit1 = self.current_slit1
            raise ValueError("slit1 out of range")
        self.port.write(int(31).to_bytes())
        ack = await self.read_bytes()
        if ack != int(31).to_bytes():
            raise SerialException("bad ack")
        # convert wavelength to 3 bytes and send
        b = int(round(self.target_slit1)).to_bytes(2)
        self.port.write(b)
        await self.read_status_end(timeout=30)

    async def go_to_slit2(self):
        """Command monochromater to go to slit2 position"""
        # must be between 10 and 3000 microns
        if self.target_slit2 > 3000 or self.target_slit2 < 10:
            self.target_slit2 = self.current_slit2
            raise ValueError("slit2 out of range")
        self.port.write(int(32).to_bytes())
        ack = await self.read_bytes()
        if ack != int(32).to_bytes():
            raise SerialException("bad ack")
        # convert wavelength to 3 bytes and send
        b = int(round(self.target_slit2)).to_bytes(2)
        self.port.write(b)
        await self.read_status_end(timeout=30)

    async def update(self):
        if self.port.is_open:
            if self.comm_up:
                # process commands from queue
                try:
                    try:
                        cmd = self.q.get_nowait()
                    except Empty:
                        # nothing in queue, get current wavelength and slits
                        self.current_wavelength = await self.get_current_wavelength()
                        self.current_slit1, self.current_slit2 = (
                            await self.get_current_slits()
                        )
                    else:
                        self.status = DkMonochromator.BUSY
                        await cmd()
                        self.status = DkMonochromator.READY
                except (SerialException, ValueError):
                    self.status = DkMonochromator.ERROR
            else:
                # wait until comm up, then do some setup
                try:
                    self.status = DkMonochromator.BUSY
                    self.comm_up = await self.establish_connection()
                    self.serial_number = await self.get_sn()
                    self.current_wavelength = await self.get_current_wavelength()
                    self.current_slit1, self.current_slit2 = (
                        await self.get_current_slits()
                    )
                    self.status = DkMonochromator.READY
                except SerialException:
                    self.status = DkMonochromator.ERROR

    def close(self):
        self.comm_up = False
        self.status = DkMonochromator.BUSY
        self.port.close()
