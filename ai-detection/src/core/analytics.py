import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.homography import HomographyHandler
from src.core.court import TennisCourt
from src.detectors.stroke import StrokeDetector


class MatchAnalyzer(BaseDetector):
    def __init__(self, stroke_model_path: Optional[str] = None, device: str = 'cpu'):
        self.court_ref = TennisCourt()
        self.homography_handler = HomographyHandler(self.court_ref)
        self.stroke_detector = None
        if stroke_model_path:
            try:
                self.stroke_detector = StrokeDetector(stroke_model_path, device=device)
            except Exception as e:
                print(f"[MatchAnalyzer] Warning: stroke model failed to load ({e}). Falling back to geometric classification.")

    def process(self, context: VideoContext) -> VideoContext:
        print("[MatchAnalyzer] Starting analysis...")

        context.analytics_data["ball_speeds"] = []
        context.analytics_data["ball_ownership"] = []
        context.analytics_data["player_stats"] = {
            "top":    {"forehands": 0, "backhands": 0, "drops": []},
            "bottom": {"forehands": 0, "backhands": 0, "drops": []},
        }

        # --- 1. Project ball track to metric space ---
        metric_track = []
        for pos, matrix in zip(context.ball_track, context.homography_matrices):
            if pos[0] is not None and matrix is not None:
                metric_track.append(self.homography_handler.project_point(pos, matrix))
            else:
                metric_track.append(None)
        player_metric_tracks = self._build_player_metric_tracks(context)

        # --- 2. Build sorted anchor list (bounces with known 2D positions) ---
        anchors = sorted(
            [a for a in context.bounce_analysis if a.get("pos_2d") is not None],
            key=lambda x: x["frame"]
        )
        if not anchors:
            print("[MatchAnalyzer] No anchors with pos_2d — skipping analytics.")
            context.analytics_data["ball_ownership"] = self._build_ball_ownership(
                context,
                metric_track,
                player_metric_tracks,
                [],
            )
            return context

        # --- 3. Bounce frames to skip when searching for hits ---
        bounce_frame_set = set(context.bounces)

        # --- 4. Build shot events, then stabilize hitter labels across the sequence ---
        shot_events = []
        for idx, anchor in enumerate(anchors):
            # --- Find hit frame (for speed calc & stroke classification) ---
            prev_anchor_frame = anchors[idx - 1]["frame"] if idx > 0 else 0
            hit_frame = self._find_hit_frame(
                anchor["frame"], prev_anchor_frame, metric_track, bounce_frame_set
            )
            if hit_frame is None:
                # Fall back to midpoint between previous bounce and this one
                hit_frame = max(prev_anchor_frame + 2, anchor["frame"] - 20)

            side, side_source, side_confidence, side_scores, side_signals = self._score_hitter_side(
                hit_frame,
                anchor["frame"],
                prev_anchor_frame,
                metric_track,
                bounce_pos_2d=anchor["pos_2d"],
                player_metric_tracks=player_metric_tracks,
            )

            # --- Speed over the hit→bounce segment ---
            speed_kmh = self._calc_segment_speed(hit_frame, anchor["frame"], metric_track, context.fps)

            shot_events.append({
                "start": hit_frame,
                "end": anchor["frame"],
                "speed_kmh": speed_kmh,
                "side": side,
                "raw_side": side,
                "side_source": side_source,
                "side_confidence": side_confidence,
                "side_scores": side_scores,
                "side_signals": side_signals,
                "bounce_pos_2d": anchor["pos_2d"],
            })

        shot_events = self._stabilize_hitter_sequence(shot_events)
        context.analytics_data["ball_ownership"] = self._build_ball_ownership(
            context,
            metric_track,
            player_metric_tracks,
            shot_events,
        )

        for event in shot_events:
            side = event["side"]
            stroke, stroke_meta = self._classify_stroke(event["start"], side, metric_track, context)
            context.analytics_data["ball_speeds"].append({
                "start": event["start"],
                "end": event["end"],
                "speed_kmh": event["speed_kmh"],
                "side": side,
                "raw_side": event["raw_side"],
                "side_source": event["side_source"],
                "side_confidence": event["side_confidence"],
                "side_scores": event["side_scores"],
                "side_signals": event["side_signals"],
                "stroke": stroke,
                "stroke_source": stroke_meta["source"],
                "stroke_confidence": stroke_meta["confidence"],
                "contact_side": stroke_meta["contact_side"],
            })
            context.analytics_data["player_stats"][side][f"{stroke}s"] += 1
            context.analytics_data["player_stats"][side]["drops"].append(event["bounce_pos_2d"])

        total = sum(
            v for side in ("top", "bottom")
            for k, v in context.analytics_data["player_stats"][side].items()
            if k in ("forehands", "backhands")
        )
        print(f"[MatchAnalyzer] Done. {len(anchors)} bounces → {len(context.analytics_data['ball_speeds'])} events, {total} strokes.")
        return context

    def _build_player_metric_tracks(self, context: VideoContext) -> Dict[str, list]:
        tracks = {"top": [None] * len(context.ball_track), "bottom": [None] * len(context.ball_track)}
        for frame_idx, frame_players in enumerate(context.players):
            if frame_idx >= len(context.homography_matrices):
                break
            matrix = context.homography_matrices[frame_idx]
            if matrix is None:
                continue
            for side in ("top", "bottom"):
                bboxes = frame_players.get(side, [])
                if not bboxes:
                    continue
                bbox = bboxes[0]
                foot = ((bbox[0] + bbox[2]) / 2, bbox[3])
                tracks[side][frame_idx] = self.homography_handler.project_point(foot, matrix)
        return tracks

    def _score_hitter_side(
        self,
        hit_frame: int,
        bounce_frame: int,
        prev_bounce_frame: int,
        metric_track: list,
        bounce_pos_2d: tuple,
        player_metric_tracks: Dict[str, list],
    ) -> tuple:
        scores = {"top": 0.0, "bottom": 0.0}
        signals = {}

        direction = self._estimate_shot_direction(hit_frame, bounce_frame, prev_bounce_frame, metric_track)
        if direction:
            side = "top" if direction["dy"] > 0 else "bottom"
            weight = 2.2 * direction["confidence"]
            scores[side] += weight
            scores[self._other_side(side)] -= 0.4 * weight
            signals["direction"] = {
                "side": side,
                "weight": round(weight, 3),
                "dy": round(direction["dy"], 4),
                "confidence": round(direction["confidence"], 3),
            }

        landing_side = "top" if bounce_pos_2d[1] > self.court_ref.config.net_y else "bottom"
        scores[landing_side] += 2.0
        scores[self._other_side(landing_side)] -= 0.4
        signals["landing_half"] = {"side": landing_side, "weight": 2.0}

        hit_pos = self._median_metric_point(metric_track, hit_frame - 3, hit_frame + 3)
        if hit_pos is not None:
            hit_side = "top" if hit_pos[1] < self.court_ref.config.net_y else "bottom"
            scores[hit_side] += 0.8
            signals["hit_location"] = {
                "side": hit_side,
                "weight": 0.8,
                "y": round(hit_pos[1], 3),
            }

            proximity = self._score_player_proximity(hit_pos, hit_frame, player_metric_tracks)
            if proximity:
                scores[proximity["side"]] += proximity["weight"]
                scores[self._other_side(proximity["side"])] -= 0.25 * proximity["weight"]
                signals["player_proximity"] = proximity

        side = "top" if scores["top"] >= scores["bottom"] else "bottom"
        margin = abs(scores["top"] - scores["bottom"])
        total = abs(scores["top"]) + abs(scores["bottom"]) + 1e-6
        confidence = max(0.0, min(1.0, margin / total))
        source = "+".join(signals.keys()) if signals else "unknown"
        return side, source, round(confidence, 3), {k: round(v, 3) for k, v in scores.items()}, signals

    def _estimate_shot_direction(self, hit_frame: int, bounce_frame: int, prev_bounce_frame: int, metric_track: list):
        start = max(prev_bounce_frame + 1, hit_frame - 4)
        end = min(bounce_frame, len(metric_track) - 1)
        samples = [(i, metric_track[i][1]) for i in range(start, end + 1) if metric_track[i] is not None]
        if len(samples) < 4:
            return None

        signed_votes = []
        window = min(5, max(2, len(samples) // 3))
        for offset in range(0, len(samples) - window):
            left = [y for _, y in samples[offset : offset + window]]
            right = [y for _, y in samples[offset + 1 : offset + 1 + window]]
            dy = float(np.median(right) - np.median(left))
            if abs(dy) > 0.025:
                signed_votes.append(dy)

        if not signed_votes:
            return None

        dy = float(np.sum(signed_votes))
        total_motion = float(np.sum(np.abs(signed_votes)))
        confidence = abs(dy) / total_motion if total_motion > 0 else 0.0
        if abs(dy) < 0.08 or confidence < 0.45:
            return None
        return {"dy": dy, "confidence": confidence}

    def _score_player_proximity(self, hit_pos: tuple, hit_frame: int, player_metric_tracks: Dict[str, list]):
        positions = {
            side: self._median_metric_point(player_metric_tracks[side], hit_frame - 6, hit_frame + 6)
            for side in ("top", "bottom")
        }
        if positions["top"] is None or positions["bottom"] is None:
            return None

        distances = {
            side: float(np.linalg.norm(np.array(hit_pos) - np.array(positions[side])))
            for side in ("top", "bottom")
        }
        side = "top" if distances["top"] <= distances["bottom"] else "bottom"
        far_side = self._other_side(side)
        gap = distances[far_side] - distances[side]
        if gap <= 0:
            return None
        confidence = min(1.0, gap / 4.0)
        weight = 1.4 * confidence
        return {
            "side": side,
            "weight": round(weight, 3),
            "distance_m": round(distances[side], 3),
            "gap_m": round(gap, 3),
            "confidence": round(confidence, 3),
        }

    def _stabilize_hitter_sequence(self, events: list) -> list:
        if len(events) < 2:
            return events

        sides = ("top", "bottom")
        transition_bonus = 0.55
        scores = []
        backrefs = []

        for idx, event in enumerate(events):
            side_scores = event.get("side_scores", {"top": 0.0, "bottom": 0.0})

            event_scores = {}
            event_backrefs = {}
            for side in sides:
                evidence_score = float(side_scores.get(side, 0.0))

                if idx == 0:
                    event_scores[side] = evidence_score
                    event_backrefs[side] = None
                    continue

                candidates = []
                for prev_side in sides:
                    transition_score = transition_bonus if prev_side != side else -transition_bonus
                    candidates.append((scores[idx - 1][prev_side] + evidence_score + transition_score, prev_side))
                event_scores[side], event_backrefs[side] = max(candidates, key=lambda item: item[0])

            scores.append(event_scores)
            backrefs.append(event_backrefs)

        final_side = max(scores[-1], key=scores[-1].get)
        sequence = [final_side]
        for idx in range(len(events) - 1, 0, -1):
            sequence.append(backrefs[idx][sequence[-1]])
        sequence.reverse()

        stabilized = [dict(event) for event in events]
        for event, side in zip(stabilized, sequence):
            if event["side"] != side:
                event["side"] = side
                event["side_source"] = f"{event['side_source']}+sequence_smooth"

        return stabilized

    def _build_ball_ownership(
        self,
        context: VideoContext,
        metric_track: list,
        player_metric_tracks: Dict[str, list],
        shot_events: list,
    ) -> list:
        contacts = []

        for event in shot_events:
            contacts.append({
                "frame": int(event["start"]),
                "side": event["side"],
                "confidence": float(event.get("side_confidence", 0.5)),
                "scores": event.get("side_scores", {"top": 0.0, "bottom": 0.0}),
                "source": f"shot_event:{event.get('side_source', 'unknown')}",
            })

        contacts.extend(self._detect_distance_contact_candidates(context, metric_track, player_metric_tracks))
        contacts = self._filter_contacts_with_bounce_anchors(contacts, shot_events)
        contacts = self._merge_contact_candidates(contacts)
        contacts = self._stabilize_contact_sequence(contacts)

        ownership = []
        contact_idx = -1
        active = None
        for frame_idx in range(len(context.ball_track)):
            while contact_idx + 1 < len(contacts) and contacts[contact_idx + 1]["frame"] <= frame_idx:
                contact_idx += 1
                active = contacts[contact_idx]

            if active is None:
                ownership.append({
                    "frame": frame_idx,
                    "owner": None,
                    "confidence": 0.0,
                    "source": "unknown_before_contact",
                    "contact_frame": None,
                })
                continue

            frames_since_contact = frame_idx - active["frame"]
            confidence = max(0.15, float(active["confidence"]) - frames_since_contact * 0.002)
            ownership.append({
                "frame": frame_idx,
                "owner": active["side"],
                "confidence": round(confidence, 3),
                "source": active["source"],
                "contact_frame": active["frame"],
            })

        return ownership

    def _detect_distance_contact_candidates(
        self,
        context: VideoContext,
        metric_track: list,
        player_metric_tracks: Dict[str, list],
        min_spacing: int = 8,
    ) -> list:
        frame_scores = []
        for frame_idx in range(len(context.ball_track)):
            score = self._score_frame_contact(context, frame_idx, metric_track, player_metric_tracks)
            frame_scores.append(score)

        candidates = []
        for frame_idx, score in enumerate(frame_scores):
            if score is None:
                continue
            side = score["side"]
            confidence = score["confidence"]
            if confidence < 0.34 or score["scores"][side] < 0.55:
                continue

            start = max(0, frame_idx - 3)
            end = min(len(frame_scores) - 1, frame_idx + 3)
            local_scores = [
                frame_scores[i]["scores"][side]
                for i in range(start, end + 1)
                if frame_scores[i] is not None
            ]
            if local_scores and score["scores"][side] < max(local_scores):
                continue

            candidate = {
                "frame": frame_idx,
                "side": side,
                "confidence": confidence,
                "scores": score["scores"],
                "source": "distance_contact",
            }
            if candidates and frame_idx - candidates[-1]["frame"] < min_spacing:
                if candidate["confidence"] > candidates[-1]["confidence"]:
                    candidates[-1] = candidate
            else:
                candidates.append(candidate)

        return candidates

    def _filter_contacts_with_bounce_anchors(self, contacts: list, shot_events: list) -> list:
        if not shot_events:
            return contacts

        filtered = []
        for contact in contacts:
            if contact["source"].startswith("shot_event"):
                filtered.append(contact)
                continue

            protected_event = None
            for event in shot_events:
                start = int(event["start"]) + 4
                end = int(event["end"]) + 2
                if start <= contact["frame"] <= end:
                    protected_event = event
                    break

            if protected_event is None:
                filtered.append(contact)
                continue

            # During a known hit->bounce flight, a descending high ball can look
            # closer to the receiver in 2D. Trust the bounce-anchored shot owner
            # unless distance evidence is extremely strong and agrees with it.
            if contact["side"] == protected_event["side"]:
                filtered.append(contact)

        return filtered

    def _score_frame_contact(
        self,
        context: VideoContext,
        frame_idx: int,
        metric_track: list,
        player_metric_tracks: Dict[str, list],
    ):
        if frame_idx >= len(context.players):
            return None
        ball = context.ball_track[frame_idx]
        if ball[0] is None:
            return None

        players = context.players[frame_idx]
        scores = {"top": 0.0, "bottom": 0.0}
        details = {}
        for side in ("top", "bottom"):
            bboxes = players.get(side, [])
            if not bboxes:
                continue
            image_score, image_detail = self._image_contact_score(ball, bboxes[0])
            metric_score = 0.0
            if metric_track[frame_idx] is not None and player_metric_tracks[side][frame_idx] is not None:
                dist_m = float(np.linalg.norm(np.array(metric_track[frame_idx]) - np.array(player_metric_tracks[side][frame_idx])))
                metric_score = float(np.exp(-dist_m / 3.5))
            scores[side] = 2.4 * image_score + 0.5 * metric_score
            details[side] = {
                **image_detail,
                "metric_score": round(metric_score, 3),
                "score": round(scores[side], 3),
            }

        if scores["top"] <= 0 and scores["bottom"] <= 0:
            return None
        side = "top" if scores["top"] >= scores["bottom"] else "bottom"
        margin = abs(scores["top"] - scores["bottom"])
        total = scores["top"] + scores["bottom"] + 1e-6
        confidence = min(1.0, margin / total)
        return {
            "side": side,
            "confidence": round(confidence, 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "details": details,
        }

    @staticmethod
    def _image_contact_score(ball: tuple, bbox) -> tuple:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        bx, by = float(ball[0]), float(ball[1])

        # Expand the box to approximate arm/racket reach. This works better for
        # airborne balls than projecting the ball to the court plane.
        ex1 = x1 - width * 0.75
        ex2 = x2 + width * 0.75
        ey1 = y1 - height * 0.35
        ey2 = y2 + height * 0.20
        closest_x = min(max(bx, ex1), ex2)
        closest_y = min(max(by, ey1), ey2)
        dist = float(np.hypot(bx - closest_x, by - closest_y))
        norm = dist / max(18.0, np.sqrt(width * height) * 0.45)
        score = float(np.exp(-(norm ** 2) / 2.0))
        inside_reach = ex1 <= bx <= ex2 and ey1 <= by <= ey2
        if inside_reach:
            score = max(score, 0.9)
        return score, {
            "image_score": round(score, 3),
            "image_dist_norm": round(norm, 3),
            "inside_reach": inside_reach,
        }

    def _merge_contact_candidates(self, contacts: list, merge_window: int = 6) -> list:
        if not contacts:
            return []

        contacts = sorted(contacts, key=lambda item: item["frame"])
        merged = [contacts[0]]
        for contact in contacts[1:]:
            prev = merged[-1]
            if contact["frame"] - prev["frame"] <= merge_window:
                prev_weight = 1.2 if prev["source"].startswith("shot_event") else 1.0
                curr_weight = 1.2 if contact["source"].startswith("shot_event") else 1.0
                prev_quality = prev["confidence"] * prev_weight
                curr_quality = contact["confidence"] * curr_weight
                if curr_quality > prev_quality:
                    merged[-1] = contact
                continue
            merged.append(contact)
        return merged

    def _stabilize_contact_sequence(self, contacts: list) -> list:
        if len(contacts) < 2:
            return contacts

        events = []
        for contact in contacts:
            side = contact["side"]
            confidence = float(contact["confidence"])
            events.append({
                "side": side,
                "side_scores": {
                    side: confidence,
                    self._other_side(side): -0.5 * confidence,
                },
                "side_source": contact["source"],
            })

        stabilized_events = self._stabilize_hitter_sequence(events)
        stabilized = [dict(contact) for contact in contacts]
        for contact, event in zip(stabilized, stabilized_events):
            if contact["side"] != event["side"]:
                contact["side"] = event["side"]
                contact["source"] = f"{contact['source']}+sequence_smooth"
        return stabilized

    @staticmethod
    def _other_side(side: str) -> str:
        return "bottom" if side == "top" else "top"

    @staticmethod
    def _median_metric_point(track: list, start_frame: int, end_frame: int):
        start = max(0, start_frame)
        end = min(len(track) - 1, end_frame)
        points = [track[i] for i in range(start, end + 1) if track[i] is not None]
        if not points:
            return None
        arr = np.array(points, dtype=np.float32)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    # -----------------------------------------------------------------------
    # Hit frame search
    # -----------------------------------------------------------------------

    def _find_hit_frame(
        self,
        bounce_frame: int,
        prev_bounce_frame: int,
        metric_track: list,
        bounce_frame_set: set,
        max_lookback: int = 60,
    ) -> Optional[int]:
        """
        Scans backward from bounce_frame looking for a kinematic direction change.
        Skips frames that are themselves bounces to avoid confusing ground-bounces
        with player hits.
        Returns the frame index of the detected hit, or None.
        """
        start = max(prev_bounce_frame + 2, bounce_frame - max_lookback)

        for i in range(bounce_frame - 4, start, -1):
            if i in bounce_frame_set:
                continue

            p0 = metric_track[i - 1] if i - 1 >= 0 else None
            p1 = metric_track[i]
            p2 = metric_track[i + 1] if i + 1 < len(metric_track) else None

            if not (p0 and p1 and p2):
                continue

            v_in  = np.array([p1[0] - p0[0], p1[1] - p0[1]])
            v_out = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            mag_in  = float(np.linalg.norm(v_in))
            mag_out = float(np.linalg.norm(v_out))

            if mag_in < 0.02 or mag_out < 0.02:
                continue

            cos_t = float(np.clip(np.dot(v_in, v_out) / (mag_in * mag_out), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_t)))

            is_hit = angle > 30 or (np.dot(v_in, v_out) < 0 and abs(v_in[1]) > 0.05)
            if is_hit:
                return i

        return None

    # -----------------------------------------------------------------------
    # Speed calculation
    # -----------------------------------------------------------------------

    def _calc_segment_speed(
        self,
        start_frame: int,
        end_frame: int,
        metric_track: list,
        fps: float,
    ) -> float:
        length = end_frame - start_frame
        if length < 5:
            return 0.0

        trim = max(1, int(length * 0.1))
        segment = [p for p in metric_track[start_frame + trim : end_frame - trim] if p is not None]
        if len(segment) < 3:
            return 0.0

        speeds = []
        for k in range(1, len(segment)):
            dx = segment[k][0] - segment[k - 1][0]
            dy = segment[k][1] - segment[k - 1][1]
            speeds.append(np.sqrt(dx * dx + dy * dy) * fps * 3.6)

        med = float(np.median(speeds))
        return med if 20 < med < 280 else 0.0

    # -----------------------------------------------------------------------
    # Stroke classification
    # -----------------------------------------------------------------------

    def _classify_stroke(
        self,
        hit_frame: int,
        side: str,
        metric_track: list,
        context: VideoContext,
    ) -> Tuple[str, Dict[str, Any]]:
        # Try ML model first
        if self.stroke_detector and 0 <= hit_frame < len(context.players):
            bboxes = context.players[hit_frame].get(side, [])
            if bboxes:
                try:
                    frame_img = context.get_frame(hit_frame)
                    stroke = self.stroke_detector.classify_stroke(frame_img, bboxes[0])
                    if stroke in ("forehand", "backhand"):
                        # Bottom player faces opposite direction → swap labels
                        if side == "bottom":
                            stroke = "backhand" if stroke == "forehand" else "forehand"
                        return stroke, {
                            "source": "stroke_model",
                            "confidence": 0.8,
                            "contact_side": None,
                        }
                except Exception:
                    pass

        # Fallback: geometric (ball position relative to player)
        ball_m = self._median_metric_point(metric_track, hit_frame - 2, hit_frame + 2)
        if ball_m is None:
            return "forehand", {
                "source": "default_missing_ball",
                "confidence": 0.0,
                "contact_side": None,
            }

        player_m = self._get_player_pos_m(hit_frame, side, context)
        if player_m is None:
            return "forehand", {
                "source": "default_missing_player",
                "confidence": 0.0,
                "contact_side": None,
            }

        lateral_delta = ball_m[0] - player_m[0]
        contact_side = "right" if lateral_delta > 0 else "left"
        confidence = min(1.0, abs(lateral_delta) / 1.2)
        if side == "top":
            stroke = "forehand" if lateral_delta < 0 else "backhand"
        else:
            stroke = "forehand" if lateral_delta > 0 else "backhand"

        if confidence < 0.15:
            stroke = "forehand"

        return stroke, {
            "source": "geometry_right_handed_assumption",
            "confidence": round(confidence, 3),
            "contact_side": contact_side,
            "lateral_delta_m": round(float(lateral_delta), 3),
        }

    def _get_player_pos_m(self, frame_idx: int, side: str, context: VideoContext) -> Optional[tuple]:
        """Returns player foot position in metric space, searching ±5 frames."""
        for win in range(max(0, frame_idx - 5), min(len(context.players), frame_idx + 6)):
            bboxes = context.players[win].get(side, [])
            if not bboxes:
                continue
            matrix = context.homography_matrices[win]
            if matrix is None:
                continue
            bbox = bboxes[0]
            foot = ((bbox[0] + bbox[2]) / 2, bbox[3])
            pos_m = self.homography_handler.project_point(foot, matrix)
            if pos_m:
                return pos_m
        return None
