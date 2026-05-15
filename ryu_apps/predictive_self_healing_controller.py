# ryu_apps/predictive_self_healing_controller.py

from collections import defaultdict, deque
from statistics import mean, pstdev
import os
import csv
import time

import joblib
import pandas as pd

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, arp

from ryu.topology import event
from ryu.topology.api import get_link


class PredictiveSelfHealingController(app_manager.RyuApp):
    """
    AI-Driven Self-Healing SDN controller for Phase 8

    Combines:
      - path forwarding
      - telemetry polling
      - live feature engineering
      - ML inference
      - predictive rerouting

    Corrected design goals:
      - OpenFlow 1.3 compatible
      - predictive healing allowed ONLY on inter-switch ports
      - temporary predictive avoidance (not permanent accumulation)
      - cooldown + consecutive threshold to reduce oscillation
      - MTTR and packet-loss related logging retained for evaluation
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # ---------------------------------------------------------
    # Flow priorities
    # ---------------------------------------------------------
    TABLE_MISS_PRIORITY = 0
    NORMAL_FLOW_PRIORITY = 50
    PREDICTIVE_FLOW_PRIORITY = 100

    FLOW_IDLE_TIMEOUT = 60
    FLOW_HARD_TIMEOUT = 0

    # ---------------------------------------------------------
    # Telemetry / prediction
    # ---------------------------------------------------------
    STATS_POLL_INTERVAL = 1.0
    FEATURE_WINDOW = 5

    DEFAULT_MODEL_PATH = "models/fault_prediction_model.pkl"
    DEFAULT_RISK_THRESHOLD = 0.50
    DEFAULT_CONSECUTIVE_POLLS = 2
    DEFAULT_COOLDOWN_SEC = 10.0
    DEFAULT_POST_HEAL_CHECK_SEC = 3.0

    # ---------------------------------------------------------
    # Logging files
    # ---------------------------------------------------------
    TELEMETRY_LOG_FILE = "results/phase8_predictive_events.csv"

    def __init__(self, *args, **kwargs):
        super(PredictiveSelfHealingController, self).__init__(*args, **kwargs)

        # ---------------------------------------------------------
        # Topology / forwarding state
        # ---------------------------------------------------------
        self.adjacency = defaultdict(dict)    # adjacency[u][v] = out_port on u toward v
        self.switch_ports = defaultdict(set)  # inter-switch ports on each switch
        self.datapaths = {}                   # dpid -> datapath
        self.host_location = {}               # mac -> (dpid, port)
        self.known_pairs = set()              # {(src_mac, dst_mac)}

        # ---------------------------------------------------------
        # Telemetry state
        # ---------------------------------------------------------
        self.last_port_stats = {}  # (dpid, port) -> counters snapshot
        self.port_feature_history = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.FEATURE_WINDOW))
        )
        self.port_risk_streak = defaultdict(int)  # (dpid, port) -> consecutive high-risk count
        self.port_last_trigger = {}               # (dpid, port) -> unix time of last healing
        self.port_last_risk = {}                  # (dpid, port) -> last risk score

        # ---------------------------------------------------------
        # Predictive avoidance state
        # ---------------------------------------------------------
        # Active predictive avoid-links are temporary and cleared after verification.
        self.active_predictive_avoid_links = set()
        self.predictive_avoid_expiry = 0.0
        self.active_heal_context = None

        # ---------------------------------------------------------
        # Config
        # ---------------------------------------------------------
        self.model_path = os.environ.get("MODEL_PATH", self.DEFAULT_MODEL_PATH).strip()
        self.risk_threshold = float(
            os.environ.get("RISK_THRESHOLD", self.DEFAULT_RISK_THRESHOLD)
        )
        self.required_consecutive_polls = int(
            os.environ.get("RISK_CONSECUTIVE_POLLS", self.DEFAULT_CONSECUTIVE_POLLS)
        )
        self.cooldown_sec = float(
            os.environ.get("HEALING_COOLDOWN_SEC", self.DEFAULT_COOLDOWN_SEC)
        )
        self.post_heal_check_sec = float(
            os.environ.get("POST_HEAL_CHECK_SEC", self.DEFAULT_POST_HEAL_CHECK_SEC)
        )
        self.fault_begin_file = os.environ.get("FAULT_BEGIN_FILE", "").strip()

        # ---------------------------------------------------------
        # Model bundle
        # ---------------------------------------------------------
        self.model = None
        self.model_feature_columns = None
        self._load_model_bundle()

        # ---------------------------------------------------------
        # Background threads
        # ---------------------------------------------------------
        self._init_csv()
        self._stats_thread = hub.spawn(self._stats_poller)
        self._post_heal_thread = hub.spawn(self._post_heal_monitor)

    # =========================================================
    # Model loading
    # =========================================================
    def _load_model_bundle(self):
        if not os.path.exists(self.model_path):
            self.logger.warning(
                "Predictive model not found at %s. Controller will keep forwarding but prediction is disabled.",
                self.model_path
            )
            return

        try:
            bundle = joblib.load(self.model_path)

            self.model = None
            self.model_feature_columns = None

            # Case 1: raw sklearn estimator directly saved
            if hasattr(bundle, "predict") or hasattr(bundle, "predict_proba"):
                self.model = bundle

            # Case 2: dict bundle saved from training script
            elif isinstance(bundle, dict):
                candidate_model_keys = [
                    "model",
                    "pipeline",
                    "best_model",
                    "selected_model",
                    "estimator",
                    "classifier",
                    "rf_model",
                    "final_model",
                ]
                for key in candidate_model_keys:
                    if key in bundle and (
                        hasattr(bundle[key], "predict") or
                        hasattr(bundle[key], "predict_proba")
                    ):
                        self.model = bundle[key]
                        break

                candidate_feature_keys = [
                    "feature_columns",
                    "features",
                    "selected_features",
                    "input_features",
                    "columns",
                ]
                for key in candidate_feature_keys:
                    if key in bundle and isinstance(bundle[key], (list, tuple)):
                        self.model_feature_columns = list(bundle[key])
                        break

            if self.model is None:
                raise ValueError(
                    f"Loaded bundle type={type(bundle)} but no usable estimator key was found"
                )

            self.logger.info(
                "Predictive model loaded successfully from %s | feature_columns=%s",
                self.model_path,
                self.model_feature_columns
            )

            if self.model_feature_columns:
                suspicious = [
                    c for c in self.model_feature_columns
                    if c in ("event_time", "time_to_event")
                ]
                if suspicious:
                    self.logger.warning(
                        "Model expects non-live columns %s. Runtime will fill them with 0.0. "
                        "For best Phase 8 behaviour, retrain the model without these fields.",
                        suspicious
                    )

        except Exception as e:
            self.logger.error(
                "Failed to load predictive model from %s: %s",
                self.model_path,
                e
            )
            self.model = None
            self.model_feature_columns = None

    # =========================================================
    # CSV logging
    # =========================================================
    def _init_csv(self):
        os.makedirs(os.path.dirname(self.TELEMETRY_LOG_FILE), exist_ok=True)
        if not os.path.exists(self.TELEMETRY_LOG_FILE):
            with open(self.TELEMETRY_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "dpid",
                    "port",
                    "risk_score",
                    "rx_drop_rate",
                    "tx_drop_rate",
                    "rx_bytes_rate",
                    "tx_bytes_rate",
                    "details",
                ])

    def _append_event_csv(
        self,
        timestamp,
        event_type,
        dpid="",
        port="",
        risk_score="",
        rx_drop_rate="",
        tx_drop_rate="",
        rx_bytes_rate="",
        tx_bytes_rate="",
        details=""
    ):
        with open(self.TELEMETRY_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                event_type,
                dpid,
                port,
                risk_score,
                rx_drop_rate,
                tx_drop_rate,
                rx_bytes_rate,
                tx_bytes_rate,
                details,
            ])

    # =========================================================
    # Switch features / table-miss
    # =========================================================
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        self.datapaths[dp.id] = dp

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=self.TABLE_MISS_PRIORITY,
            match=match,
            instructions=inst
        )
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

    def delete_eth_dst_flow(self, dp, dst_mac, priority=None):
        if priority is None:
            priority = self.NORMAL_FLOW_PRIORITY

        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_dst=dst_mac)

        mod = parser.OFPFlowMod(
            datapath=dp,
            command=ofp.OFPFC_DELETE,
            out_port=ofp.OFPP_ANY,
            out_group=ofp.OFPG_ANY,
            priority=priority,
            match=match
        )
        dp.send_msg(mod)

    # =========================================================
    # Topology discovery
    # =========================================================
    @set_ev_cls(event.EventSwitchEnter)
    def on_switch_enter(self, ev):
        self.build_topology()

    @set_ev_cls(event.EventLinkAdd)
    def on_link_add(self, ev):
        self.build_topology()

    @set_ev_cls(event.EventLinkDelete)
    def on_link_delete(self, ev):
        """
        Reactive fallback:
        If a real topology failure happens, rebuild topology and reroute all known pairs.
        Reactive fallback ignores predictive avoid-links so the controller can restore
        any physically available path.
        """
        now = time.time()
        self.logger.warning("FAILURE DETECTED: EventLinkDelete react_unix=%.6f", now)
        self._log_reaction_delay_if_fault_file(now)

        self.build_topology()
        self.recover_all_known_pairs(
            reason="reactive_link_delete",
            react_time=now,
            predictive=False,
            avoid_links=None
        )

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def on_port_status(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        reason = msg.reason
        desc = msg.desc

        if reason == ofp.OFPPR_MODIFY and (desc.state & ofp.OFPPS_LINK_DOWN):
            now = time.time()
            self.logger.warning(
                "FAILURE DETECTED: Port down react_unix=%.6f dpid=%s port_no=%s",
                now, dp.id, desc.port_no
            )
            self._log_reaction_delay_if_fault_file(now)

            self.build_topology()
            self.recover_all_known_pairs(
                reason="reactive_port_down",
                react_time=now,
                predictive=False,
                avoid_links=None
            )

    def build_topology(self):
        self.adjacency.clear()
        self.switch_ports.clear()

        links = get_link(self, None)
        for link in links:
            u = link.src.dpid
            v = link.dst.dpid
            port_u_to_v = link.src.port_no

            self.adjacency[u][v] = port_u_to_v
            self.switch_ports[u].add(port_u_to_v)

        self.logger.info("Topology updated: %s", dict(self.adjacency))
        self.logger.info(
            "Inter-switch ports: %s",
            {k: sorted(list(v)) for k, v in self.switch_ports.items()}
        )

    # =========================================================
    # Shortest path with optional avoidance
    # =========================================================
    def shortest_path(self, src_sw, dst_sw, avoid_links=None):
        if avoid_links is None:
            avoid_links = set()

        if src_sw == dst_sw:
            return [src_sw]

        if src_sw not in self.adjacency:
            return None

        visited = {src_sw}
        q = deque([(src_sw, [src_sw])])

        while q:
            current, path = q.popleft()
            for neigh in self.adjacency[current].keys():
                if (current, neigh) in avoid_links:
                    continue
                if neigh in visited:
                    continue
                visited.add(neigh)
                new_path = path + [neigh]
                if neigh == dst_sw:
                    return new_path
                q.append((neigh, new_path))
        return None

    def get_out_port_for_path(self, current_sw, path, dst_sw, final_port):
        if current_sw == dst_sw:
            return final_port

        if current_sw not in path:
            return None

        idx = path.index(current_sw)
        if idx >= len(path) - 1:
            return final_port

        next_sw = path[idx + 1]
        return self.adjacency.get(current_sw, {}).get(next_sw)

    def _current_avoid_links(self):
        now = time.time()
        if self.active_predictive_avoid_links and now < self.predictive_avoid_expiry:
            return set(self.active_predictive_avoid_links)
        return set()

    def _clear_predictive_avoidance(self):
        self.active_predictive_avoid_links.clear()
        self.predictive_avoid_expiry = 0.0

    # =========================================================
    # Flow installation / recomputation
    # =========================================================
    def install_path_flows(self, dst_mac, path, final_port, priority):
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
                dp=dp,
                priority=priority,
                match=match,
                actions=actions,
                idle_timeout=self.FLOW_IDLE_TIMEOUT,
                hard_timeout=self.FLOW_HARD_TIMEOUT
            )

    def recompute_and_push(self, src_mac, dst_mac, avoid_links=None, priority=None):
        if priority is None:
            priority = self.NORMAL_FLOW_PRIORITY

        if src_mac not in self.host_location or dst_mac not in self.host_location:
            return False

        src_sw, _ = self.host_location[src_mac]
        dst_sw, dst_port = self.host_location[dst_mac]

        path = self.shortest_path(src_sw, dst_sw, avoid_links=avoid_links)
        if not path:
            return False

        for dp in self.datapaths.values():
            self.delete_eth_dst_flow(dp, dst_mac, priority=self.NORMAL_FLOW_PRIORITY)
            self.delete_eth_dst_flow(dp, dst_mac, priority=self.PREDICTIVE_FLOW_PRIORITY)

        self.install_path_flows(
            dst_mac=dst_mac,
            path=path,
            final_port=dst_port,
            priority=priority
        )
        return True

    def recover_all_known_pairs(self, reason, react_time, predictive=False, avoid_links=None):
        if not self.known_pairs:
            self.logger.warning(
                "RECOVERY ACTION: reason=%s react_unix=%.6f predictive=%s (no known pairs yet)",
                reason, react_time, predictive
            )
            return

        ok = 0
        fail = 0
        prio = self.PREDICTIVE_FLOW_PRIORITY if predictive else self.NORMAL_FLOW_PRIORITY

        for (src, dst) in list(self.known_pairs):
            r1 = self.recompute_and_push(src, dst, avoid_links=avoid_links, priority=prio)
            r2 = self.recompute_and_push(dst, src, avoid_links=avoid_links, priority=prio)
            if r1 and r2:
                ok += 1
            else:
                fail += 1

        avoid_display = sorted(list(avoid_links)) if avoid_links else []
        self.logger.warning(
            "RECOVERY ACTION: reason=%s react_unix=%.6f predictive=%s pairs_ok=%d pairs_fail=%d avoid_links=%s",
            reason, react_time, predictive, ok, fail, avoid_display
        )

    # =========================================================
    # Packet-in forwarding
    # =========================================================
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        in_port = msg.match.get("in_port")
        dpid = dp.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        if eth.ethertype == 0x88CC:
            return

        src = eth.src
        dst = eth.dst

        # Learn host only on edge ports
        if in_port is not None and in_port not in self.switch_ports.get(dpid, set()):
            old_loc = self.host_location.get(src)
            new_loc = (dpid, in_port)
            self.host_location[src] = new_loc
            if old_loc != new_loc:
                self.logger.info(
                    "Host learned: mac=%s -> (dpid=%s, port=%s)",
                    src, dpid, in_port
                )

        active_avoid_links = self._current_avoid_links()

        # ARP handling
        if pkt.get_protocol(arp.arp):
            if src in self.host_location and dst in self.host_location:
                src_sw, _ = self.host_location[src]
                dst_sw, dst_host_port = self.host_location[dst]
                path = self.shortest_path(src_sw, dst_sw, avoid_links=active_avoid_links)
                if path:
                    out_port = self.get_out_port_for_path(dpid, path, dst_sw, dst_host_port)
                    if out_port is not None:
                        self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(out_port)])
                        return

            self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(ofp.OFPP_FLOOD)])
            return

        # Unknown hosts
        if dst not in self.host_location or src not in self.host_location:
            self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(ofp.OFPP_FLOOD)])
            return

        self.known_pairs.add((src, dst))

        src_sw, src_host_port = self.host_location[src]
        dst_sw, dst_host_port = self.host_location[dst]

        path = self.shortest_path(src_sw, dst_sw, avoid_links=active_avoid_links)
        if not path:
            self._packet_out(dp, msg, in_port, [parser.OFPActionOutput(ofp.OFPP_FLOOD)])
            return

        active_priority = (
            self.PREDICTIVE_FLOW_PRIORITY if active_avoid_links else self.NORMAL_FLOW_PRIORITY
        )

        self.install_path_flows(
            dst_mac=dst,
            path=path,
            final_port=dst_host_port,
            priority=active_priority
        )
        reverse_path = list(reversed(path))
        self.install_path_flows(
            dst_mac=src,
            path=reverse_path,
            final_port=src_host_port,
            priority=active_priority
        )

        out_port = self.get_out_port_for_path(dpid, path, dst_sw, dst_host_port)
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

    # =========================================================
    # Telemetry polling
    # =========================================================
    def _stats_poller(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_port_stats(dp)
            hub.sleep(self.STATS_POLL_INTERVAL)

    def _request_port_stats(self, dp):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        req = parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY)
        dp.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        dp = ev.msg.datapath
        body = ev.msg.body
        now = time.time()

        for stat in body:
            port_no = stat.port_no

            # Skip special OpenFlow ports
            if port_no > 0xFF00:
                continue

            key = (dp.id, port_no)
            snapshot = {
                "rx_bytes": stat.rx_bytes,
                "tx_bytes": stat.tx_bytes,
                "rx_packets": stat.rx_packets,
                "tx_packets": stat.tx_packets,
                "rx_dropped": stat.rx_dropped,
                "tx_dropped": stat.tx_dropped,
                "rx_errors": stat.rx_errors,
                "tx_errors": stat.tx_errors,
                "ts": now,
            }

            prev = self.last_port_stats.get(key)
            self.last_port_stats[key] = snapshot

            if prev is None:
                continue

            dt = now - prev["ts"]
            if dt <= 0:
                continue

            rx_bytes_rate = max(0.0, (snapshot["rx_bytes"] - prev["rx_bytes"]) / dt)
            tx_bytes_rate = max(0.0, (snapshot["tx_bytes"] - prev["tx_bytes"]) / dt)
            rx_packets_rate = max(0.0, (snapshot["rx_packets"] - prev["rx_packets"]) / dt)
            tx_packets_rate = max(0.0, (snapshot["tx_packets"] - prev["tx_packets"]) / dt)
            rx_drop_rate = max(0.0, (snapshot["rx_dropped"] - prev["rx_dropped"]) / dt)
            tx_drop_rate = max(0.0, (snapshot["tx_dropped"] - prev["tx_dropped"]) / dt)
            rx_error_rate = max(0.0, (snapshot["rx_errors"] - prev["rx_errors"]) / dt)
            tx_error_rate = max(0.0, (snapshot["tx_errors"] - prev["tx_errors"]) / dt)

            feature_row = {
                "timestamp": now,
                "dpid": dp.id,
                "port": port_no,
                "rx_bytes_rate": rx_bytes_rate,
                "tx_bytes_rate": tx_bytes_rate,
                "rx_packets_rate": rx_packets_rate,
                "tx_packets_rate": tx_packets_rate,
                "rx_drop_rate": rx_drop_rate,
                "tx_drop_rate": tx_drop_rate,
                "rx_error_rate": rx_error_rate,
                "tx_error_rate": tx_error_rate,
            }

            self._update_feature_history(dp.id, port_no, feature_row)
            live_features = self._build_live_features(dp.id, port_no, feature_row)

            self.logger.info(
                "LOSS_LOG: ts=%.6f dpid=%s port=%s rx_drop_rate=%.6f tx_drop_rate=%.6f "
                "rx_bytes_rate=%.2f tx_bytes_rate=%.2f",
                now, dp.id, port_no, rx_drop_rate, tx_drop_rate, rx_bytes_rate, tx_bytes_rate
            )

            if self.model is None:
                continue

            risk_score = self._predict_risk(live_features)
            self.port_last_risk[key] = risk_score

            self._append_event_csv(
                timestamp=now,
                event_type="telemetry",
                dpid=dp.id,
                port=port_no,
                risk_score=risk_score,
                rx_drop_rate=rx_drop_rate,
                tx_drop_rate=tx_drop_rate,
                rx_bytes_rate=rx_bytes_rate,
                tx_bytes_rate=tx_bytes_rate,
                details=""
            )

            self._evaluate_risk_and_heal(dp.id, port_no, risk_score, live_features)

    # =========================================================
    # Feature engineering
    # =========================================================
    def _update_feature_history(self, dpid, port_no, feature_row):
        hist = self.port_feature_history[(dpid, port_no)]

        for k in [
            "rx_bytes_rate", "tx_bytes_rate",
            "rx_packets_rate", "tx_packets_rate",
            "rx_drop_rate", "tx_drop_rate",
            "rx_error_rate", "tx_error_rate",
        ]:
            hist[k].append(feature_row[k])

    def _safe_mean(self, seq):
        return mean(seq) if seq else 0.0

    def _safe_std(self, seq):
        return pstdev(seq) if len(seq) >= 2 else 0.0

    def _safe_slope(self, seq):
        n = len(seq)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = self._safe_mean(seq)

        num = 0.0
        den = 0.0
        for i, y in enumerate(seq):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx

        if den == 0:
            return 0.0
        return num / den

    def _build_live_features(self, dpid, port_no, current):
        hist = self.port_feature_history[(dpid, port_no)]

        total_bytes_rate = current["rx_bytes_rate"] + current["tx_bytes_rate"]
        total_packets_rate = current["rx_packets_rate"] + current["tx_packets_rate"]
        total_drop_rate = current["rx_drop_rate"] + current["tx_drop_rate"]
        total_error_rate = current["rx_error_rate"] + current["tx_error_rate"]

        rx_seq = list(hist["rx_bytes_rate"])
        tx_seq = list(hist["tx_bytes_rate"])
        byte_total_seq = [a + b for a, b in zip(rx_seq, tx_seq)]

        pkt_rx_seq = list(hist["rx_packets_rate"])
        pkt_tx_seq = list(hist["tx_packets_rate"])
        packet_total_seq = [a + b for a, b in zip(pkt_rx_seq, pkt_tx_seq)]

        drop_seq = [
            a + b for a, b in zip(list(hist["rx_drop_rate"]), list(hist["tx_drop_rate"]))
        ]
        err_seq = [
            a + b for a, b in zip(list(hist["rx_error_rate"]), list(hist["tx_error_rate"]))
        ]

        features = {
            # Base live rates
            "rx_bytes_rate": current["rx_bytes_rate"],
            "tx_bytes_rate": current["tx_bytes_rate"],
            "rx_packets_rate": current["rx_packets_rate"],
            "tx_packets_rate": current["tx_packets_rate"],
            "rx_drop_rate": current["rx_drop_rate"],
            "tx_drop_rate": current["tx_drop_rate"],
            "rx_error_rate": current["rx_error_rate"],
            "tx_error_rate": current["tx_error_rate"],

            # Offline-model aligned totals
            "bytes_rate_total": total_bytes_rate,
            "packets_rate_total": total_packets_rate,
            "drop_rate_total": total_drop_rate,
            "error_rate_total": total_error_rate,

            # Ratios
            "drop_to_packet_ratio": (
                total_drop_rate / total_packets_rate if total_packets_rate > 0 else 0.0
            ),
            "error_to_packet_ratio": (
                total_error_rate / total_packets_rate if total_packets_rate > 0 else 0.0
            ),

            # Rolling means
            "bytes_rate_total_mean": self._safe_mean(byte_total_seq),
            "packets_rate_total_mean": self._safe_mean(packet_total_seq),
            "drop_rate_total_mean": self._safe_mean(drop_seq),
            "error_rate_total_mean": self._safe_mean(err_seq),

            # Rolling std
            "bytes_rate_total_std": self._safe_std(byte_total_seq),
            "packets_rate_total_std": self._safe_std(packet_total_seq),
            "drop_rate_total_std": self._safe_std(drop_seq),

            # Diffs
            "bytes_rate_total_diff": (
                byte_total_seq[-1] - byte_total_seq[-2] if len(byte_total_seq) >= 2 else 0.0
            ),
            "packets_rate_total_diff": (
                packet_total_seq[-1] - packet_total_seq[-2] if len(packet_total_seq) >= 2 else 0.0
            ),
            "drop_rate_total_diff": (
                drop_seq[-1] - drop_seq[-2] if len(drop_seq) >= 2 else 0.0
            ),
            "error_rate_total_diff": (
                err_seq[-1] - err_seq[-2] if len(err_seq) >= 2 else 0.0
            ),

            # Extremes
            "bytes_rate_total_max": max(byte_total_seq) if byte_total_seq else 0.0,
            "bytes_rate_total_min": min(byte_total_seq) if byte_total_seq else 0.0,
            "drop_rate_total_max": max(drop_seq) if drop_seq else 0.0,

            # Instability / lag
            "bytes_instability": self._safe_std(byte_total_seq),
            "drop_instability": self._safe_std(drop_seq),
            "bytes_rate_total_lag1": byte_total_seq[-2] if len(byte_total_seq) >= 2 else 0.0,
            "drop_rate_total_lag1": drop_seq[-2] if len(drop_seq) >= 2 else 0.0,
            "error_rate_total_lag1": err_seq[-2] if len(err_seq) >= 2 else 0.0,

            # Extra live-only convenience fields
            "utilisation_now": total_bytes_rate,
            "drop_rate_now": total_drop_rate,
            "error_rate_now": total_error_rate,
            "byte_rate_slope_5": self._safe_slope(byte_total_seq),

            # Present only for compatibility with flawed old models
            "event_time": 0.0,
            "time_to_event": 0.0,
        }

        return features

    # =========================================================
    # ML inference
    # =========================================================
    def _predict_risk(self, feature_map):
        try:
            if self.model_feature_columns:
                row_dict = {
                    col: float(feature_map.get(col, 0.0))
                    for col in self.model_feature_columns
                }
                X = pd.DataFrame([row_dict], columns=self.model_feature_columns)
            else:
                ordered_keys = sorted(feature_map.keys())
                row_dict = {k: float(feature_map[k]) for k in ordered_keys}
                X = pd.DataFrame([row_dict], columns=ordered_keys)

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                if len(proba[0]) >= 2:
                    return float(proba[0][1])
                return float(proba[0][0])

            pred = self.model.predict(X)
            return float(pred[0])

        except Exception as e:
            self.logger.error("Prediction failed: %s", e)
            return 0.0

    # =========================================================
    # Healing logic
    # =========================================================
    def _is_inter_switch_port(self, dpid, port_no):
        return port_no in self.switch_ports.get(dpid, set())

    def _evaluate_risk_and_heal(self, dpid, port_no, risk_score, live_features):
        key = (dpid, port_no)
        now = time.time()

        # Ignore host-facing ports for predictive topology healing.
        if not self._is_inter_switch_port(dpid, port_no):
            self.port_risk_streak[key] = 0
            return

        # If a predictive avoidance session is already active, do not keep stacking
        # new risky links on top of it. Let verification/cooldown finish first.
        if self.active_predictive_avoid_links and now < self.predictive_avoid_expiry:
            return

        last_trigger = self.port_last_trigger.get(key)
        if last_trigger is not None and (now - last_trigger) < self.cooldown_sec:
            return

        if risk_score >= self.risk_threshold:
            self.port_risk_streak[key] += 1
        else:
            self.port_risk_streak[key] = 0
            return

        self.logger.warning(
            "PREDICTION_ALERT: ts=%.6f dpid=%s port=%s risk=%.6f streak=%d threshold=%.2f",
            now, dpid, port_no, risk_score, self.port_risk_streak[key], self.risk_threshold
        )

        if self.port_risk_streak[key] < self.required_consecutive_polls:
            return

        self.port_risk_streak[key] = 0
        self.port_last_trigger[key] = now

        risky_links = self._infer_links_from_switch_port(dpid, port_no)
        if not risky_links:
            self.logger.warning(
                "Predictive trigger fired for dpid=%s port=%s but no inter-switch link mapping found. "
                "No topology avoidance applied.",
                dpid, port_no
            )
            return

        # Temporary predictive avoidance replaces any previous predictive avoidance.
        self.active_predictive_avoid_links = set(risky_links)
        self.predictive_avoid_expiry = now + self.post_heal_check_sec + self.cooldown_sec

        self.active_heal_context = {
            "trigger_time": now,
            "dpid": dpid,
            "port": port_no,
            "risk_score": risk_score,
            "drop_rate_now": live_features.get("drop_rate_now", 0.0),
            "utilisation_now": live_features.get("utilisation_now", 0.0),
            "risky_links": sorted(list(risky_links)),
        }

        self.logger.warning(
            "HEALING_ACTION: proactive_reroute trigger_unix=%.6f dpid=%s port=%s risk=%.6f risky_links=%s",
            now, dpid, port_no, risk_score, sorted(list(risky_links))
        )

        self._append_event_csv(
            timestamp=now,
            event_type="healing_action",
            dpid=dpid,
            port=port_no,
            risk_score=risk_score,
            rx_drop_rate=live_features.get("rx_drop_rate", 0.0),
            tx_drop_rate=live_features.get("tx_drop_rate", 0.0),
            rx_bytes_rate=live_features.get("rx_bytes_rate", 0.0),
            tx_bytes_rate=live_features.get("tx_bytes_rate", 0.0),
            details=f"avoid_links={sorted(list(risky_links))}"
        )

        self.recover_all_known_pairs(
            reason="predictive_reroute",
            react_time=now,
            predictive=True,
            avoid_links=set(self.active_predictive_avoid_links)
        )

        self._log_reaction_delay_if_fault_file(now)

    def _infer_links_from_switch_port(self, dpid, port_no):
        """
        Map (dpid, port_no) -> set of directed graph edges to avoid.
        Only inter-switch links are returned.
        """
        risky = set()

        if not self._is_inter_switch_port(dpid, port_no):
            return risky

        for u, nbrs in self.adjacency.items():
            if u != dpid:
                continue
            for v, out_port in nbrs.items():
                if out_port == port_no:
                    risky.add((u, v))
                    risky.add((v, u))
        return risky

    # =========================================================
    # Post-healing verification
    # =========================================================
    def _post_heal_monitor(self):
        while True:
            now = time.time()

            if self.active_heal_context is not None:
                trigger_time = self.active_heal_context["trigger_time"]
                if (now - trigger_time) >= self.post_heal_check_sec:
                    dpid = self.active_heal_context["dpid"]
                    port = self.active_heal_context["port"]
                    key = (dpid, port)

                    latest_risk = self.port_last_risk.get(key, 0.0)
                    prev_drop = self.active_heal_context.get("drop_rate_now", 0.0)

                    latest_features = self._latest_live_snapshot(key)
                    latest_drop = latest_features.get("drop_rate_now", 0.0)
                    latest_util = latest_features.get("utilisation_now", 0.0)

                    improved = latest_drop <= prev_drop

                    self.logger.warning(
                        "POST_HEAL_CHECK: trigger_unix=%.6f verify_unix=%.6f dpid=%s port=%s "
                        "prev_drop=%.6f latest_drop=%.6f latest_risk=%.6f latest_util=%.2f improved=%s",
                        trigger_time, now, dpid, port, prev_drop, latest_drop, latest_risk, latest_util, improved
                    )

                    self._append_event_csv(
                        timestamp=now,
                        event_type="post_heal_check",
                        dpid=dpid,
                        port=port,
                        risk_score=latest_risk,
                        rx_drop_rate="",
                        tx_drop_rate="",
                        rx_bytes_rate="",
                        tx_bytes_rate="",
                        details=(
                            f"prev_drop={prev_drop:.6f};"
                            f"latest_drop={latest_drop:.6f};"
                            f"improved={improved};"
                            f"clearing_predictive_avoidance=True"
                        )
                    )

                    # Clear predictive avoidance after verification so the topology
                    # does not get poisoned by permanent accumulated avoid-links.
                    self._clear_predictive_avoidance()
                    self.active_heal_context = None

            hub.sleep(1.0)

    def _latest_live_snapshot(self, key):
        dpid, port_no = key
        hist = self.port_feature_history.get((dpid, port_no), None)
        if not hist:
            return {}

        def latest_or_zero(name):
            seq = list(hist[name])
            return seq[-1] if seq else 0.0

        rx = latest_or_zero("rx_bytes_rate")
        tx = latest_or_zero("tx_bytes_rate")
        rd = latest_or_zero("rx_drop_rate")
        td = latest_or_zero("tx_drop_rate")

        return {
            "utilisation_now": rx + tx,
            "drop_rate_now": rd + td
        }

    # =========================================================
    # MTTR hook
    # =========================================================
    def _log_reaction_delay_if_fault_file(self, react_time):
        """
        Optional evaluation hook:
        if FAULT_BEGIN_FILE is exported and points to a unix timestamp file,
        log controller reaction delay vs the manually recorded fault time.
        """
        if not self.fault_begin_file:
            return

        try:
            with open(self.fault_begin_file, "r", encoding="utf-8") as f:
                fault_begin = float(f.read().strip())

            self.logger.warning(
                "MTTR_LOG_HOOK: fault_begin_unix=%.6f controller_event_unix=%.6f controller_delay=%.6f_sec",
                fault_begin, react_time, (react_time - fault_begin)
            )
        except Exception as e:
            self.logger.error(
                "Could not read FAULT_BEGIN_FILE=%s: %s",
                self.fault_begin_file, e
            )
