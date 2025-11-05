#!/usr/bin/env python3
import logging
import time
import j1939
import asyncio
import os
# from PyQt5 import QtCore, QtWidgets, QtGui
# from PyQt5.QtWidgets import QFileDialog, QMessageBox
from asyncqt import QEventLoop
import configparser
import can
from can.interfaces.pcan.basic import PCAN_DEVICE_NUMBER
from can.interfaces.pcan.pcan import PcanBus
from j1939.message_id import FrameFormat
import cantools
from pprint import pprint

def on_message(priority, pgn, sa, timestamp, data):
    """Receive incoming messages from the bus

    :param int priority:
        Priority of the message
    :param int pgn:
        Parameter Group Number of the message
    :param int sa:
        Source Address of the message
    :param int timestamp:
        Timestamp of the message
    :param bytearray data:
        Data of the PDU
    """
    print("PGN {} length {}".format(pgn, len(data)), timestamp)
    
# connect to bus
def bus_connect(device_number=None):
    """Connect to  first available pcan channel
    Baud: 500k/2M
    :param int device_number: connect to specific pcan device id. If the device Id is not found, None is returned
    :return: if successful, a can-bus object is returned, otherwise None
    :rtype: (bus-object or None)
    """
    if os.name == 'posix':
        try:
            # set socketcan link up if required
            if 'state UP' not in os.popen('ip link show can0').read():
                os.system('sudo ip link set can0 up type can bitrate 500000')

            return can.interface.Bus(bustype='socketcan', channel='can0', bitrate=500000)
        except Exception as e:
            print('can error', e)
            return None
    else:
        PcanChannels = ['PCAN_USBBUS' + str(cnt) for cnt in range(1, 17)]

        # connect to first available channel
        for channel in PcanChannels:
            try:
                # get pcan device number
                pcan = can.interface.Bus(interface='pcan', channel=channel, state=can.bus.BusState.PASSIVE, bitrate=500000,)
                number = pcan.m_objPCANBasic.GetValue(pcan.m_PcanHandle, PCAN_DEVICE_NUMBER)[1]
                pcan.shutdown()

                # try to connect in silent mode on specific device
                if device_number != None:
                    if device_number != number:
                        print('Peak interface with device-id 0x{:04x} found'.format(number), '(required device-id: ' + hex(device_number) + ')')
                        continue

                # can
                bus = can.interface.Bus(interface='pcan', channel=channel, state=can.bus.BusState.ACTIVE, fd=True, f_clock_mhz=80, nom_brp=10, nom_tseg1=12, nom_tseg2=3, nom_sjw=1, data_brp=4, data_tseg1=7, data_tseg2=2, data_sjw=1)
                print('Successfully connected to peak interface with device-id', hex(number), 'and channel', channel, '\n')
                return bus
            except Exception as e:
                print(e)
                pass
        return None

def output(hb_01_duty, hb_01_dir, hb_02_duty, hb_02_dir, hs_01_pwm, hs_02_pwm, os_01_control, os_02_control):
    # qm h-bridge
    h_bridge_write_signals['h_bridge_01_duty'] = hb_01_duty
    h_bridge_write_signals['h_bridge_01_direction'] = hb_01_dir
       
    h_bridge_write_signals['h_bridge_02_duty'] = hb_02_duty
    h_bridge_write_signals['h_bridge_02_direction'] = hb_02_dir
    # qm high-side
    high_side_write_signals['high_side_out_01_value_pwm'] = hs_01_pwm
    high_side_write_signals['high_side_out_02_value_pwm'] = hs_02_pwm
    # fusa output
    fusa_output_control_signals['ch_01_ctrl_val'] = os_01_control
    fusa_output_control_signals['ch_02_ctrl_val'] = os_02_control
    send_messages()    

def send_message(msg, signals):
    data = msg.encode(signals)

    can_message = can.Message(arbitration_id=msg.frame_id, data=data, is_fd=True, bitrate_switch=True)
    bus.send(can_message)


def send_messages():
    send_message(h_bridge_write_msg, h_bridge_write_signals)
    send_message(high_side_write_msg, high_side_write_signals)
    send_message(fusa_output_control_msg, fusa_output_control_signals)

def main():

    print("python-can version", can.__version__)
    print("can-j1939 version", j1939.__version__)

    # create the ElectronicControlUnit (one ECU can hold multiple ControllerApplications) with j1939-22 data link layer
    ecu = j1939.ElectronicControlUnit(data_link_layer='j1939-22', max_cmdt_packets=200)
    pcan_device_id = None
    try:
        config = configparser.ConfigParser()
        config.read('pcan-config.ini')
        pcan_device_id = int(config.get('pcan', 'device_id' ))
    except:
        pass
        
    ecu._bus = bus_connect(pcan_device_id)
    if ecu._bus == None:
        print( "Error: No pcan-adapter found!" )
    
    ecu._notifier = can.Notifier(ecu._bus, ecu._listeners, 1)

    # subscribe to all (global) messages on the bus
    ecu.subscribe(on_message)

    # compose the name descriptor for the new ca
    name = j1939.Name(
        arbitrary_address_capable=0,
        industry_group=j1939.Name.IndustryGroup.Industrial,
        vehicle_system_instance=1,
        vehicle_system=1,
        function=1,
        function_instance=1,
        ecu_instance=1,
        manufacturer_code=666,
        identity_number=1234567
        )
    
    # create the ControllerApplications
    ca = j1939.ControllerApplication(name, 0x1)
    ecu.add_ca(controller_application=ca)
    ca.start()

    # init h-bridge init message
    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_01')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_02')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_03')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_04')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_05')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_06')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    h_bridge_init_msg = db.get_message_by_name('QM_H_bridge_init_07')
    h_bridge_init_signals = {}
    for signal in h_bridge_init_msg.signals:
        if signal.initial == None:
            h_bridge_init_signals[signal.name] = 0
        else:
            h_bridge_init_signals[signal.name] = signal.initial

    # init h-bridge write message
    h_bridge_write_msg = db.get_message_by_name('QM_H_bridge_write')
    h_bridge_write_signals = {}
    for signal in h_bridge_write_msg.signals:
        if signal.initial == None:
            h_bridge_write_signals[signal.name] = 0
        else:
            h_bridge_write_signals[signal.name] = signal.initial

    # init high-side-01 - 03 init message
    high_side_init_01_msg = db.get_message_by_name('QM_High_side_output_init_01')
    high_side_init_01_signals = {}
    for signal in high_side_init_01_msg.signals:
        if signal.initial == None:
            high_side_init_01_signals[signal.name] = 0
        else:
            high_side_init_01_signals[signal.name] = signal.initial

    high_side_init_02_msg = db.get_message_by_name('QM_High_side_output_init_02')
    high_side_init_02_signals = {}
    for signal in high_side_init_02_msg.signals:
        if signal.initial == None:
            high_side_init_02_signals[signal.name] = 0
        else:
            high_side_init_02_signals[signal.name] = signal.initial

    high_side_init_03_msg = db.get_message_by_name('QM_High_side_output_init_03')
    high_side_init_03_signals = {}
    for signal in high_side_init_03_msg.signals:
        if signal.initial == None:
            high_side_init_03_signals[signal.name] = 0
        else:
            high_side_init_03_signals[signal.name] = signal.initial

    # init high-side write message
    high_side_write_msg = db.get_message_by_name('QM_High_side_output_write')
    high_side_write_signals = {}
    for signal in high_side_write_msg.signals:
        if signal.initial == None:
            high_side_write_signals[signal.name] = 0
        else:
            high_side_write_signals[signal.name] = signal.initial

    # send init messages
    send_message(h_bridge_init_msg,     h_bridge_init_signals)
    send_message(high_side_init_01_msg, high_side_init_01_signals)
    send_message(high_side_init_02_msg, high_side_init_02_signals)
    send_message(high_side_init_03_msg, high_side_init_03_signals)

    time.sleep(1000)
    ca.stop()
    ecu.disconnect()


if __name__ == "__main__":
    main()
    while True:
        pass