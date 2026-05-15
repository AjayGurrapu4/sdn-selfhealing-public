# ryu_apps/reactive_fault_baseline.py

from collections import defaultdict, deque
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, arp

from ryu.topology import event
from ryu.topology.api import get_link


class ReactiveFaultBaseline(app_manager.RyuApp):
    """
    Phase 3 — Reactive Fault Management Baseline (comparison system)

    Detect failure:
      - EventLinkDelete (Ryu topology)
      - EventOFPPortStatus when OFPPS_LINK_DOWN is set

    Recover:
      - rebuild adjacency graph
      - recompute shortest path
      - delete old eth_dst flows (priority FLOW_PRIORITY)
      - install new path flows

    Logging (evaluation support):
      - Controller reaction timestamps
      - Packet loss counters/rates from PortStats polling
      - Optional: if FAULT_BEGIN_FILE is set, log reaction delay from fault begin
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    FLOW_PRIORITY = 50
    FLOW_IDLE_TIMEOUT = 60

    STATS_POLL_INTERVAL = 1.0  # seconds

    def __init__(self, *args, **kwargs):
        super(ReactiveFaultBaseline, self).__init__(*args, **kwargs)

        # adjacency[u][v] = port_on_u_that_reaches_v
        self.adjacency = defaultdict(dict)

        # datapaths[dpid] = datapath object
        self.datapaths = {}

        # host_location[mac] = (dpid, port)
        self.host_location = {}

        # host pairs we have installed paths for
        self.known_pairs = set()

        # For packet-loss logging (PortStats deltas)
        # last_port_stats[(dpid, port_no)] = dict(counters..., ts=...)
        self.last_port_stats = {}

        # Optional fault begin file (written by your trigger script)
        self.fault_begin_file = os.environ.get("FAULT_BEGIN_FILE", "").strip()

        # background stats poller
        self._stats_thread = hub.spawn(self._stats_poller)

    # -------------------------
    # Switch connect: table-miss
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        self.datapaths[dp.id] = dp

        # Table-miss -> controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)

        self.logger.info("Switch connected: dpid=%s (table-miss installed)", dp.id)

    def add_flow(self, dp, priority, match, actions, idle_timeout=60, hard_timeout=0):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        dp.send_msg(mod)

    def delete_eth_dst_flow(self, dp, dst_mac):
        """
        Delete only our path flows for eth_dst=dst_mac at FLOW_PRIORITY.
        """
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_dst=dst_mac)
        mod = parser.OFPFlowMod(
            datapath=dp,
            command=ofp.OFPFC_DELETE,
            out_port=ofp.OFPP_ANY,
            out_group=ofp.OFPG_ANY,
            priority=self.FLOW_PRIORITY,
            match=match
        )
        dp.send_msg(mod)

    # -------------------------
    # Topology build/update
    # -------------------------
    def build_topology(self):
        self.adjacency.clear()
        links = get_link(self, None)
        for link in links:
            u = link.src.dpid
            v = link.dst.dpid
            port_u_to_v = link.src.port_no
            self.adjacency[u][v] = port_u_to_v

        self.logger.info("Topology rebuilt (adjacency): %s", dict(self.adjacency))

    @set_ev_cls(event.EventSwitchEnter)
    def on_switch_enter(self, ev):
        self.build_topology()

    @set_ev_cls(event.EventLinkAdd)
    def on_link_add(self, ev):
        self.build_topology()

    # -------------------------
    # Failure detection
    # -------------------------
    @set_ev_cls(event.EventLinkDelete)
    def on_link_delete(self, ev):
        react_time = time.time()
        self.logger.warning("FAILURE DETECTED: EventLinkDelete react_unix=%.6f", react_time)
        self._log_reaction_delay_if_fault_file(react_time)

        self.build_topology()
        self.recover_all_known_pairs(reason="link_delete", react_time=react_time)

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def on_port_status(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        reason = msg.reason
        desc = msg.desc

        # We care about MODIFY + link-down bit
        if reason == ofp.OFPPR_MODIFY:
            if desc.state & ofp.OFPPS_LINK_DOWN:
                react_time = time.time()
                self.logger.warning(
                    "FAILURE DETECTED: Port down react_unix=%.6f dpid=%s port_no=%s",
                    react_time, dp.id, desc.port_no
                )
                self._log_reaction_delay_if_fault_file(react_time)

                self.build_topology()
                self.recover_all_known_pairs(reason="port_down", react_time=react_time)

    def _log_reaction_delay_if_fault_file(self, react_time):
        """
        Optional: if you export FAULT_BEGIN_FILE=/path/to/fault_begin.txt
        we compute controller reaction delay = react_time - fault_begin.
        """
        if not self.fault_begin_file:
            return
        try:
            with open(self.fault_begin_file, "r", encoding="utf-8") as f:
                s = f.read().strip()
            fault_begin = float(s)
            self.logger.warning(
                "MTTR_LOG_HOOK: fault_begin_unix=%.6f react_unix=%.6f controller_react_delay=%.6f_sec",
                fault_begin, react_time, (react_time - fault_begin)
            )
        except Exception as e:
            self.logger.error("Could not read FAULT_BEGIN_FILE=%s: %s", self.fault_begin_file, e)

    # -------------------------
    # Shortest path (BFS)
    # -------------------------
    def shortest_path(self, src_sw, dst_sw):
        if src_sw == dst_sw:
            return [src_sw]
        if src_sw not in self.adjacency:
            return None

        visited = set([src_sw])
        q = deque([(src_sw, [src_sw])])

        while q:
            cur, path = q.popleft()
            for neigh in self.adjacency[cur].keys():
                if neigh in visited:
                    continue
                visited.add(neigh)
                new_path = path + [neigh]
                if neigh == dst_sw:
                    return new_path
                q.append((neigh, new_path))
        return None

    # -------------------------
    # Flow install / recompute
    # -------------------------
    def install_path_flows(self, dst_mac, path, final_port):
        """
        Install eth_dst=dst_mac path flows along 'path'.
        """
        for i, sw in enumerate(path):
            dp = self.datapaths.get(sw)
            if not dp:
                continue

            parser = dp.ofproto_parser

            if i == len(path) - 1:
                out_port = final_port
            else:
                next_sw = path[i + 1]
                out_port = self.adjacency.get(sw, {}).get(next_sw)

            if out_port is None:
                continue

            match = parser.OFPMatch(eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(
                dp,
                priority=self.FLOW_PRIORITY,
                match=match,
                actions=actions,
                idle_timeout=self.FLOW_IDLE_TIMEOUT
            )

    def recompute_and_push(self, src_mac, dst_mac):
        """
        Recompute path for src->dst and push new flows for dst_mac.
        Returns True on success.
        """
        if src_mac not in self.host_location or dst_mac not in self.host_location:
            return False

        src_sw, _ = self.host_location[src_mac]
        dst_sw, dst_port = self.host_location[dst_mac]

        path = self.shortest_path(src_sw, dst_sw)
        if not path:
            return False

        # delete old dst flows everywhere (clean baseline)
        for dp in self.datapaths.values():
            self.delete_eth_dst_flow(dp, dst_mac)

        # install new dst flows along the recomputed path
        self.install_path_flows(dst_mac=dst_mac, path=path, final_port=dst_port)
        return True

    def recover_all_known_pairs(self, reason, react_time):
        """
        Proactive recovery: reroute all known host pairs immediately,
        instead of waiting for next PacketIn.
        """
        if not self.known_pairs:
            self.logger.warning(
                "RECOVERY ACTION: reason=%s react_unix=%.6f (no known pairs yet)",
                reason, react_time
            )
            return

        ok = 0
        fail = 0
        for (src, dst) in list(self.known_pairs):
            r1 = self.recompute_and_push(src, dst)
            r2 = self.recompute_and_push(dst, src)
            if r1 and r2:
                ok += 1
            else:
                fail += 1

        self.logger.warning(
            "RECOVERY ACTION: reason=%s react_unix=%.6f pairs_ok=%d pairs_fail=%d",
            reason, react_time, ok, fail
        )

    # -------------------------
    # Packet-in: learn hosts + install paths
    # -------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        in_port = msg.match.get('in_port')
        dpid = dp.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        # ignore LLDP
        if eth.ethertype == 0x88cc:
            return

        src = eth.src
        dst = eth.dst

        # Learn host attachment
        if in_port is not None:
            self.host_location[src] = (dpid, in_port)

        # ARP: flood (needed for host discovery)
        if pkt.get_protocol(arp.arp):
            self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(ofp.OFPP_FLOOD)])
            return

        # unknown dst host: flood until learned
        if dst not in self.host_location:
            self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(ofp.OFPP_FLOOD)])
            return

        # record pair for proactive recovery later
        self.known_pairs.add((src, dst))

        src_sw, _ = self.host_location[src]
        dst_sw, dst_port = self.host_location[dst]

        path = self.shortest_path(src_sw, dst_sw)
        if not path:
            self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(ofp.OFPP_FLOOD)])
            return

        # install forward & reverse paths
        self.recompute_and_push(src, dst)
        self.recompute_and_push(dst, src)

        # forward this packet along the chosen path
        out_port = None
        if dpid == dst_sw:
            out_port = dst_port
        else:
            if dpid in path:
                idx = path.index(dpid)
                if idx < len(path) - 1:
                    next_sw = path[idx + 1]
                    out_port = self.adjacency.get(dpid, {}).get(next_sw)

        if out_port is None:
            out_port = ofp.OFPP_FLOOD

        self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(out_port)])

    def _packet_out(self, dp, msg, in_port, actions):
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        if msg.buffer_id != ofp.OFP_NO_BUFFER:
            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=None
            )
        else:
            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=ofp.OFP_NO_BUFFER,
                in_port=in_port,
                actions=actions,
                data=msg.data
            )
        dp.send_msg(out)

    # -------------------------
    # Port stats polling (packet loss logging)
    # -------------------------
    def _stats_poller(self):
        while True:
            try:
                for dp in list(self.datapaths.values()):
                    self._send_port_stats_request(dp)
            except Exception as e:
                self.logger.error("Stats poller error: %s", e)
            hub.sleep(self.STATS_POLL_INTERVAL)

    def _send_port_stats_request(self, dp):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        req = parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY)
        dp.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def on_port_stats_reply(self, ev):
        dp = ev.msg.datapath
        now = time.time()

        for stat in ev.msg.body:
            # Skip LOCAL port noise if needed
            port_no = int(stat.port_no)
            key = (dp.id, port_no)

            rx_packets = int(stat.rx_packets)
            tx_packets = int(stat.tx_packets)
            rx_dropped = int(stat.rx_dropped)
            tx_dropped = int(stat.tx_dropped)
            rx_errors = int(stat.rx_errors)
            tx_errors = int(stat.tx_errors)

            prev = self.last_port_stats.get(key)
            self.last_port_stats[key] = {
                "ts": now,
                "rx_packets": rx_packets,
                "tx_packets": tx_packets,
                "rx_dropped": rx_dropped,
                "tx_dropped": tx_dropped,
                "rx_errors": rx_errors,
                "tx_errors": tx_errors,
            }

            if not prev:
                continue

            dt = now - prev["ts"]
            if dt <= 0:
                continue

            # deltas
            drx_drop = rx_dropped - prev["rx_dropped"]
            dtx_drop = tx_dropped - prev["tx_dropped"]
            drx_err = rx_errors - prev["rx_errors"]
            dtx_err = tx_errors - prev["tx_errors"]

            # loss rate per second (simple baseline metric)
            drop_rate = (drx_drop + dtx_drop) / dt
            err_rate = (drx_err + dtx_err) / dt

            # Log occasionally (you can grep these later)
            # Keep it readable; this is your "packet loss logging" evidence.
            if (drx_drop + dtx_drop + drx_err + dtx_err) > 0:
                self.logger.warning(
                    "LOSS_LOG dpid=%s port=%s drop_rate=%.3fpps err_rate=%.3fpps "
                    "rx_drop=%d tx_drop=%d rx_err=%d tx_err=%d",
                    dp.id, port_no, drop_rate, err_rate,
                    rx_dropped, tx_dropped, rx_errors, tx_errors
                )

