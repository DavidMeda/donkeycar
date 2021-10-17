"""
pins.py - high level ttl an pwm pin abstraction.
This is designed to allow drivers that use ttl and pwm
pins to be reusable with different underlying libaries
and techologies.

The abstract classes InputPin, OutputPin and PwmPin 
provide an interface for starting, using and cleaning up the pins.
The factory functions input_pin_by_id(), output_pin_by_id()
and pwm_pin_by_id() construct pins given a string id
that specifies the underlying pin provider and it's attributes.
There are implementations for the Rpi.GPIO library and
for the PCA9685.  

Pin id allows pins to be selected using a single string to
select from different underlying providers, numbering schemes and settings.

Use Rpi.GPIO library, GPIO.BOARD pin numbering scheme, pin number 13
 pin = input_pin_by_id("RPI_GPIO.BOARD.13")

Use Rpi.GPIO library, GPIO.BCM broadcom pin numbering scheme, gpio pin number 33
 pin = output_pin_by_id("RPI_GPIO.BCM.33")

Use PCA9685 on bus 0 at address 0x40, channel 7
 pin = pwm_pin_by_id("PCA9685.0:40.7")

TODO: implement PiGPIO pin provider
"""
from abc import ABC, abstractmethod

from donkeycar.parts import  actuator
import RPi.GPIO as GPIO


class PinState:
    LOW:int = 0
    HIGH:int = 1
    NOT_STARTED:int = -1


class PinEdge:
    RISING:int = 1
    FALLING:int = 2
    BOTH:int = 3


class PinPull:
    PULL_NONE:int = 1
    PULL_UP:int = 2
    PULL_DOWN:int = 3


class PinProvider:
    RPI_GPIO = "RPI_GPIO"
    PCA9685 = "PCA9685"
    # PIGPIO = "PIGPIO"


class PinScheme:
    BOARD = "BOARD"  # board numbering
    BCM = "BCM"      # broadcom gpio numbering


# 
##### Base interface for input/output/pwm pins
##### Implementations derive from these abstact classes
#

class InputPin(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def start(self, on_input=None, edge=PinEdge.RISING) -> None:
        """
        Start the pin in input mode.
        on_input: function to call when an edge is detected, or None to ignore
        edge: type of edge(s) that trigger on_input; default is PinEdge.RISING
        This raises a RuntimeError if the pin is already started.
        You can check to see if the pin is started by calling
        state() and checking for PinState.NOT_STARTED
        """
        pass  # subclasses should override this

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the pin and return it to PinState.NOT_STARTED
        """
        pass  # subclasses should override this

    @abstractmethod
    def state(self) -> int:
        """
        Return most recent input state.  This does not re-read the input pin,
        it just returns that last value read by the input() method.
        If the pin is not started or has been stopped, 
        this will return PinState:NOT_STARTED
        """
        return PinState.NOT_STARTED  # subclasses must override

    @abstractmethod
    def input(self) -> int:
        """
        Read the input state from the pin.
        """
        return PinState.NOT_STARTED  # subclasses must override


class OutputPin(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def start(self, state:int=PinState.LOW) -> None:
        """
        Start the pin in output mode and with given starting state.
        This raises and RuntimeError if the pin is already started.
        You can check to see if the pin is started by calling
        state() and checking for PinState.NOT_STARTED
        """
        pass  # subclasses should override this

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the pin and return it to PinState.NOT_STARTED
        """
        pass  # subclasses should override this

    @abstractmethod
    def state(self) -> int:
        """
        Return most recent output state.  This does not re-read the pin,
        It just returns that last value set by the output() method.
        If the pin is not started or has been stopped, 
        this will return PinState:NOT_STARTED
        """
        return PinState.NOT_STARTED  # subclasses must override

    @abstractmethod
    def output(self, state:int) -> None:
        """
        Set the output state of the pin to either
        PinState.LOW or PinState.HIGH
        """
        pass  # subclasses must override
        

class PwmPin(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def start(self, state:float=0) -> None:
        """
        Start the pin in output mode and with given starting state.
        This raises and RuntimeError if the pin is already started.
        You can check to see if the pin is started by calling
        state() and checking for PinState.NOT_STARTED
        """
        pass  # subclasses should override this

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the pin and return it to PinState.NOT_STARTED
        """
        pass  # subclasses should override this

    @abstractmethod
    def state(self) -> float:
        """
        Return most recent output state.  This does not re-read the pin,
        It just returns that last value set by the output() method.
        If the pin is not started or has been stopped, 
        this will return PinState:NOT_STARTED
        """
        return PinState.NOT_STARTED  # subclasses must override

    @abstractmethod
    def duty_cycle(self, state:float) -> None:
        """
        Set the output duty cycle of the pin 
        in range 0 to 1.0 (0% to 100%)
        """
        pass  # subclasses must override


#
######## Factory Methods
#


#
# Pin id allows pins to be selected using a single string to
# select from different underlying providers, numbering schemes and settings.
#
# Use Rpi.GPIO library, GPIO.BOARD pin numbering scheme, pin number 13
# "RPI_GPIO.BOARD.13"
#
# Use Rpi.GPIO library, GPIO.BCM broadcom pin numbering scheme, gpio pin number 33
# "RPI_GPIO.BCM.33"
#
# Use PCA9685 on bus 0 at address 0x40, channel 7
# "PCA9685.0:40.7"
#
def output_pin_by_id(pin_id:str, frequency_hz:int=60) -> OutputPin:
    """
    Select a ttl output pin given a pin id.
    """
    parts = pin_id.split(".")
    if parts[0] == PinProvider.PCA9685:
        pin_provider = parts[0]
        i2c_bus, i2c_address = parts[1].split(":")
        i2c_bus = int(i2c_bus)
        i2c_address = int(i2c_address, base=16)
        frequency_hz = int(frequency_hz)
        pin_number = int(parts[2])
        return output_pin(pin_provider, pin_number, i2c_bus=i2c_bus, i2c_address=i2c_address, frequency_hz=frequency_hz)

    if parts[0] == PinProvider.RPI_GPIO:
        pin_provider = parts[0]
        pin_scheme = parts[1]
        pin_number = int(parts[2])
        return output_pin(pin_provider, pin_number, pin_scheme=pin_scheme)


def pwm_pin_by_id(pin_id:str, frequency_hz:int=60) -> OutputPin:
    """
    Select a pwm output pin given a pin id.
    """
    parts = pin_id.split(".")
    if parts[0] == PinProvider.PCA9685:
        pin_provider = parts[0]
        i2c_bus, i2c_address = parts[1].split(":")
        i2c_bus = int(i2c_bus)
        i2c_address = int(i2c_address, base=16)
        pin_number = int(parts[2])
        return pwm_pin(pin_provider, pin_number, i2c_bus=i2c_bus, i2c_address=i2c_address, frequency_hz=frequency_hz)

    if parts[0] == PinProvider.RPI_GPIO:
        pin_provider = parts[0]
        pin_scheme = parts[1]
        pin_number = int(parts[2])
        return pwm_pin(pin_provider, pin_number, pin_scheme=pin_scheme, frequency_hz=frequency_hz)


def input_pin_by_id(pin_id:str, pull:int=PinPull.PULL_NONE) -> OutputPin:
    """
    Select a ttl input pin given a pin id.
    """
    parts = pin_id.split(".")
    if parts[0] == PinProvider.PCA9685:
        raise RuntimeError("PinProvider.PCA9685 does not implement InputPin")

    if parts[0] == PinProvider.RPI_GPIO:
        pin_provider = parts[0]
        pin_scheme = parts[1]
        pin_number = int(parts[2])
        return input_pin(pin_provider, pin_number, pin_scheme=pin_scheme, pull=pull)


def input_pin(pin_provider:str, pin_number:int, pin_scheme:str=PinScheme.BOARD, pull:int=PinPull.PULL_NONE) -> InputPin:
    """
    construct an InputPin using the given pin provider
    """
    if pin_provider == PinProvider.RPI_GPIO:
        return InputPinGpio(pin_number, pin_scheme, pull)
    if pin_provider == PinProvider.PCA9685:
        raise RuntimeError("PinProvider.PCA9685 does not implement InputPin")
    raise RuntimeError("UnknownPinProvider ({})".format(pin_provider))


def output_pin(pin_provider:str, pin_number:int, pin_scheme:str=PinScheme.BOARD, i2c_bus:int=0, i2c_address:int=40, frequency_hz:int=60) -> OutputPin:
    """
    construct an output pin using the given pin provider
    """
    if pin_provider == PinProvider.RPI_GPIO:
        return OutputPinGpio(pin_number, pin_scheme)
    if pin_provider == PinProvider.PCA9685:
        return OutputPinPCA9685(pin_number, frequency_hz, i2c_bus, i2c_address)
    raise RuntimeError("UnknownPinProvider ({})".format(pin_provider))


def pwm_pin(pin_provider:str, pin_number:int, pin_scheme:str=PinScheme.BOARD, frequency_hz:int=60, i2c_bus:int=0, i2c_address:int=40) -> PwmPin:
    """
    construct a PwmPin using the given pin provider
    """
    if pin_provider == PinProvider.RPI_GPIO:
        return PwmPinGpio(pin_number, pin_scheme, frequency_hz)
    if pin_provider == PinProvider.PCA9685:
        return PwmPinPCA9685(pin_number, frequency_hz, i2c_bus, i2c_address)
    raise RuntimeError("UnknownPinProvider ({})".format(pin_provider))


#
# RPi.GPIO/Jetson.GPIO implementations
#
def gpio_fn(pin_scheme, fn):
    """
    Convenience method to call GPIO function
    using desired pin scheme.  This restores
    the previous pin scheme, so it is safe to 
    mix pin schemes.
    """
    prev_scheme = GPIO.getmode() or pin_scheme
    GPIO.setmode(pin_scheme)
    val = fn()
    GPIO.setmode(prev_scheme)
    return val


# lookups to convert abstact api to GPIO values
gpio_pin_edge = [GPIO.RISING, GPIO.FALLING, GPIO.BOTH]
gpio_pin_pull = [GPIO.PUD_OFF, GPIO.PUD_DOWN, GPIO.PUD_UP]
gpio_pin_scheme = {PinScheme.BOARD: GPIO.BOARD, PinScheme.BCM: GPIO.BCM}


class InputPinGpio(InputPin):
    def __init__(self, pin_number:int, pin_scheme:str, pull=PinPull.PULL_NONE) -> None:
        """
        Input pin ttl HIGH/LOW using RPi.GPIO/Jetson.GPIO
        pin_number: GPIO.BOARD mode point number
        pull: enable a pull up or down resistor on pin.  Default is PinPull.PULL_NONE
        """
        self.pin_number = pin_number
        self.pin_scheme = gpio_pin_scheme[pin_scheme]
        self.pull = pull
        self._state = PinState.NOT_STARTED
        super().__init__()

    def start(self, on_input=None, edge=PinEdge.RISING) -> None:
        """
        on_input: function to call when an edge is detected, or None to ignore
        edge: type of edge(s) that trigger on_input; default is 
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError("Attempt to start InputPin that is already started.")
        gpio_fn(self.pin_scheme, lambda: GPIO.setup(self.pin_number, GPIO.IN, pull_up_down=gpio_pin_pull(self.pull)))
        if on_input is not None:
            gpio_fn(self.pin_scheme, lambda: GPIO.add_event_detect(self.pin_number, gpio_pin_edge(edge), callback=on_input))
        self.input()  # read first state

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
            self._state = PinState.NOT_STARTED

    def state(self) -> int:
        return self._state

    def input(self) -> int:
        self._state = gpio_fn(self.pin_scheme, lambda: GPIO.input(self.pin_number))
        return self._state


class OutputPinGpio(OutputPin):
    """
    Output pin ttl HIGH/LOW using Rpi.GPIO/Jetson.GPIO
    """
    def __init__(self, pin_number:int, pin_scheme:str) -> None:
        self.pin_number = pin_number
        self.pin_scheme = gpio_pin_scheme[pin_scheme]
        self._state = PinState.NOT_STARTED

    def start(self, state:int=PinState.LOW) -> None:
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError("Attempt to start OutputPin that is already started.")
        gpio_fn(self.pin_scheme, lambda: GPIO.setup(self.pin_number, GPIO.OUT))
        self.output(state)

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
            self._state = PinState.NOT_STARTED

    def state(self) -> int:
        return self._state

    def output(self, state: int) -> None:
        gpio_fn(self.pin_scheme, lambda: GPIO.output(self.pin_number, state))
        self._state = state


class PwmPinGpio(PwmPin):
    """
    PWM output pin using Rpi.GPIO/Jetson.GPIO
    """
    def __init__(self, pin_number:int, pin_scheme:str, frequency_hz = 50) -> None:
        self.pin_number = pin_number
        self.pin_scheme = gpio_pin_scheme[pin_scheme]
        self.frequency = frequency_hz
        self.pwm = None
        self._state = PinState.NOT_STARTED

    def start(self, duty:float=0) -> None:
        if self.pwm is not None:
            raise RuntimeError("Attempt to start PwmPinGpio that is already started.")
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle must be in range 0 to 1")
        gpio_fn(self.pin_scheme, lambda: GPIO.setup(self.pin_number, GPIO.OUT))
        self.pwm = gpio_fn(self.pin_scheme, lambda: GPIO.PWM(self.pin_number, self.frequency))
        self.duty_cycle(duty)
        self._state = duty

    def stop(self) -> None:
        if self.pwm is not None:
            self.pwm.stop()
            gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
        self._state = PinState.NOT_STARTED

    def state(self) -> float:
        return self._state

    def duty_cycle(self, duty: float) -> None:
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle must be in range 0 to 1")
        self.pwm.ChangeDutyCycle(duty * 100)
        self._state = duty_cycle


#
# PCA9685 implementations
# 
class OutputPinPCA9685(ABC):
    """
    Output pin ttl HIGH/LOW using PCA9685
    """
    def __init__(self, pin_number:int, frequency_hz:int, i2c_bus:int, i2c_address:int) -> None:
        self.pin_number = pin_number
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.frequency_hz = frequency_hz
        self.pca9685 = None
        self._state = PinState.NOT_STARTED

    def start(self, state:int=PinState.LOW) -> None:
        """
        Start the pin in output mode.
        This raises and RuntimeError if the pin is already started.
        You can check to see if the pin is started by calling
        state() and checking for PinState.NOT_STARTED
        """
        if self.pca9685 is not None:
            raise RuntimeError("Attempt to start pin ({}) that is already started".format(self.pin_number))
        self.pca9685 = actuator.PCA9685(self.pin_number, self.i2c_address, self.frequency_hz, self.i2c_bus)
        self.output(state)

    def stop(self) -> None:
        """
        Stop the pin and return it to PinState.NOT_STARTED
        """
        if self.pca9685 is not None:
            self.output(PinState.LOW)
            self.pca9685 = None
        self._state = PinState.NOT_STARTED

    def state(self) -> int:
        """
        Return most recent output state.  
        If the pin is not started or has been stopped, 
        this will return PinState:NOT_STARTED
        """
        return self._state

    def output(self, state: int) -> None:
        self.pca9685.set_pulse(1 if state == PinState.HIGH else 0)
        self._state = state


class PwmPinPCA9685(PwmPin):
    """
    PWM output pin using PCA9685
    """
    def __init__(self, pin_number:int, frequency_hz:int, i2c_bus:int, i2c_address:int) -> None:
        self.pin_number = pin_number
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.frequency_hz = frequency_hz
        self.pca9685 = None
        self._state = PinState.NOT_STARTED

    def start(self, duty:float=0) -> None:
        if self.pca9685 is not None:
            raise RuntimeError("Attempt to start pin ({}) that is already started".format(self.pin_number))
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle must be in range 0 to 1")
        self.pca9685 = actuator.PCA9685(self.pin_number, self.i2c_address, self.frequency_hz, self.i2c_bus)
        self.duty_cycle(duty)
        self._state = duty

    def stop(self) -> None:
        if self.pca9685 is not None:
            self.duty_cycle(0)
            self.pca9685 = None
        self._state = PinState.NOT_STARTED

    def state(self) -> float:
        return self._state

    def duty_cycle(self, duty: float) -> None:
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle must be in range 0 to 1")
        self.pca9685.set_duty_cycle(duty)
        self._state = duty
