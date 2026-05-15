#!/usr/bin/env python3

import os
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel, info


class DiamondTopo(Topo):
    def build(self):
        # ---- Hosts ----
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        # ---- Switches ----
        s1 = self.addSwitch('s1')  # edge (h1 side)
        s2 = self.addSwitch('s2')  # core upper
        s3 = self.addSwitch('s3')  # core lower
        s4 = self.addSwitch('s4')  # edge (h2 side)

        # ---- Links: host-edge ----
        self.addLink(h1, s1, cls=TCLink, bw=100, delay='1ms')
        self.addLink(h2, s4, cls=TCLink, bw=100, delay='1ms')

        # ---- Links: diamond redundancy ----
        # Upper path: s1-s2-s4
        self.addLink(s1, s2, cls=TCLink, bw=50, delay='5ms')
        self.addLink(s2, s4, cls=TCLink, bw=50, delay='5ms')

        # Lower path: s1-s3-s4
        self.addLink(s1, s3, cls=TCLink, bw=50, delay='5ms')
        self.addLink(s3, s4, cls=TCLink, bw=50, delay='5ms')


def run():
    topo = DiamondTopo()

    # Controller (Ryu) running locally on TCP/6633
    c0 = RemoteController('c0', ip='127.0.0.1', port=6633)

    net = Mininet(
        topo=topo,
        controller=c0,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=False
    )

    info("** Starting network\n")
    net.start()

    # Force OpenFlow 1.3
    info("** Forcing OpenFlow13 on all switches\n")
    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set bridge {sw.name} protocols=OpenFlow13')

    # STP OPTIONAL:
    # For Phase 3 reroute baseline, usually STP disabled so both branches are usable.
    enable_stp = os.environ.get("ENABLE_STP", "0") == "1"
    if enable_stp:
        info("** Enabling STP (ENABLE_STP=1)\n")
        for sw in net.switches:
            sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=true')
    else:
        info("** STP disabled (default). Both diamond branches remain available.\n")
        for sw in net.switches:
            sw.cmd(f'ovs-vsctl set bridge {sw.name} stp_enable=false')

    CLI(net)

    info("** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()

